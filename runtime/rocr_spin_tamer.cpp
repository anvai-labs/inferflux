#include "runtime/rocr_spin_tamer.h"

#include "runtime/string_utils.h"
#include "runtime/thread_name.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#if defined(__linux__)
#include <dirent.h>
#include <dlfcn.h>
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <ucontext.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#endif

namespace inferflux {
namespace {

#if defined(__linux__)
const char *kComponent = "rocr_tamer";

constexpr int kWindowMs = 1200;
constexpr int kSleepSliceMs = 50;
constexpr int kMinRunnablePercent = 80;
constexpr long kMaxStimeDeltaTicks = 2;
// PC proof: at least kPcOwnedRequired of kPcSamples samples must land in the
// code owner (the rest may be vDSO/libc helpers the spin loop calls without
// a syscall). Sampling stops as soon as the verdict is decided, so a
// non-owner receives at most 2 probe signals per attempt.
constexpr int kPcSamples = 3;
constexpr int kPcOwnedRequired = 2;
// Generous: a runnable-but-starved spinner on a loaded host may wait on a
// runqueue before the handler runs. Scans are background work.
constexpr int kPcReplyTimeoutMs = 500;
// Upper bound for an in-flight handler to finish after its probe ended.
constexpr int kHandlerDrainTimeoutMs = 1000;
// A thread not proven waits 1, 2, 4, ... scans (capped) before re-probing.
constexpr unsigned long long kMaxBackoffScans = 12;
// The ROCr runtime spawns 2 pollers; caps blast radius if anything misfires.
// Counts threads confined and still alive, so a runtime that is torn down
// and re-initialized (new pollers) never exhausts it.
constexpr std::size_t kMaxConfinedAlive = 8;
// Offset from SIGRTMIN for the PC probe. glibc reserves the RT signals below
// its SIGRTMIN; nothing in InferFlux or llama.cpp/ggml uses RT signals, and
// a foreign handler (at install time or any later scan) disables probing.
constexpr int kProbeSignalOffset = kRocrTamerProbeSignalOffset;

using ThreadKey = std::pair<pid_t, unsigned long long>; // (tid, starttime)

struct ThreadSample {
  unsigned long long starttime = 0; // Distinguishes a reused TID.
  long utime_ticks = 0;
  long stime_ticks = 0;
  long long voluntary_switches = 0;
  // From /proc/.../schedstat (CONFIG_SCHED_INFO): ns on CPU and ns runnable
  // but waiting on a runqueue. Absent on kernels without schedstats.
  bool has_schedstat = false;
  unsigned long long run_ns = 0;
  unsigned long long wait_ns = 0;
};

std::string TaskPath(pid_t tid, const char *leaf) {
  return "/proc/self/task/" + std::to_string(tid) + "/" + leaf;
}

std::vector<pid_t> ListThreads() {
  std::vector<pid_t> tids;
  DIR *dir = opendir("/proc/self/task");
  if (!dir) {
    return tids;
  }
  while (dirent *entry = readdir(dir)) {
    char *end = nullptr;
    const long tid = std::strtol(entry->d_name, &end, 10);
    if (end && *end == '\0' && tid > 0) {
      tids.push_back(static_cast<pid_t>(tid));
    }
  }
  closedir(dir);
  return tids;
}

// comm may contain spaces/parens, so parse fields after the LAST ')'. Field
// N of the stat format is tokens[N - 3]: utime(14), stime(15),
// starttime(22).
bool ReadThreadStat(pid_t tid, ThreadSample *out) {
  std::ifstream file(TaskPath(tid, "stat"));
  std::string line;
  if (!file || !std::getline(file, line)) {
    return false; // TID vanished mid-sample; caller skips it.
  }
  const std::size_t close = line.rfind(')');
  if (close == std::string::npos) {
    return false;
  }
  std::istringstream fields(line.substr(close + 1));
  std::vector<std::string> tokens;
  std::string token;
  while (fields >> token) {
    tokens.push_back(token);
  }
  if (tokens.size() < 20) {
    return false;
  }
  out->utime_ticks = std::strtol(tokens[11].c_str(), nullptr, 10);
  out->stime_ticks = std::strtol(tokens[12].c_str(), nullptr, 10);
  out->starttime = std::strtoull(tokens[19].c_str(), nullptr, 10);
  return true;
}

// Reads one "Key:\tvalue" line of /proc/.../status. strtoull skips the
// separator whitespace itself, so this does not depend on /proc padding.
bool ReadStatusField(pid_t tid, const char *key, int base,
                     unsigned long long *out) {
  std::ifstream file(TaskPath(tid, "status"));
  const std::size_t key_len = std::strlen(key);
  std::string line;
  while (file && std::getline(file, line)) {
    if (line.compare(0, key_len, key) == 0) {
      *out = std::strtoull(line.c_str() + key_len, nullptr, base);
      return true;
    }
  }
  return false; // TID vanished; caller skips it.
}

// Reads several "Key:" fields of /proc/.../status in one pass. Returns
// false unless every key was found.
bool ReadStatusFields(pid_t tid,
                      std::initializer_list<std::pair<const char *, int>> keys,
                      unsigned long long *out) {
  std::ifstream file(TaskPath(tid, "status"));
  std::size_t found = 0;
  std::string line;
  while (found < keys.size() && file && std::getline(file, line)) {
    std::size_t i = 0;
    for (const auto &[key, base] : keys) {
      const std::size_t key_len = std::strlen(key);
      if (line.compare(0, key_len, key) == 0) {
        out[i] = std::strtoull(line.c_str() + key_len, nullptr, base);
        ++found;
      }
      ++i;
    }
  }
  return found == keys.size();
}

// The scheduler state letter (stat field 3): 'R' = running or runnable.
char ReadThreadState(pid_t tid) {
  std::ifstream file(TaskPath(tid, "stat"));
  std::string line;
  if (!file || !std::getline(file, line)) {
    return '\0';
  }
  const std::size_t close = line.rfind(')');
  return close != std::string::npos && close + 2 < line.size() ? line[close + 2]
                                                               : '\0';
}

bool ReadSchedstat(pid_t tid, unsigned long long *run_ns,
                   unsigned long long *wait_ns) {
  std::ifstream file(TaskPath(tid, "schedstat"));
  return static_cast<bool>(file >> *run_ns >> *wait_ns);
}

std::unordered_map<pid_t, ThreadSample> SampleAllThreads() {
  std::unordered_map<pid_t, ThreadSample> samples;
  for (const pid_t tid : ListThreads()) {
    ThreadSample sample;
    unsigned long long voluntary = 0;
    if (!ReadThreadStat(tid, &sample) ||
        !ReadStatusField(tid, "voluntary_ctxt_switches:", 10, &voluntary)) {
      continue;
    }
    sample.voluntary_switches = static_cast<long long>(voluntary);
    sample.has_schedstat = ReadSchedstat(tid, &sample.run_ns, &sample.wait_ns);
    samples.emplace(tid, sample);
  }
  return samples;
}

// O(1) fast path: is the ROCr runtime loaded at all? RTLD_NOLOAD never loads
// anything; it only returns a handle when the library is already mapped.
bool HsaRuntimeLoaded() {
  for (const char *soname : {"libhsa-runtime64.so.1", "libhsa-runtime64.so"}) {
    if (void *handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD)) {
      dlclose(handle);
      return true;
    }
  }
  return false;
}

// --- Verdict memory --------------------------------------------------------
// Keyed by (tid, starttime) so a reused TID starts fresh; entries for
// threads no longer alive are pruned every scan, bounding the memory.
struct Backoff {
  unsigned long long strikes = 0;
  unsigned long long skip_through_scan = 0; // Skip scans <= this sequence.
};

std::mutex g_memory_mutex;
std::set<ThreadKey> g_confined;
std::map<ThreadKey, Backoff> g_backoff;
unsigned long long g_scan_seq = 0;

// Starts a scan: advances the sequence and prunes entries for threads that
// no longer exist (exited, or TID reused). Returns the scan's sequence.
unsigned long long
BeginScan(const std::unordered_map<pid_t, ThreadSample> &alive) {
  const auto is_dead = [&alive](const ThreadKey &key) {
    const auto it = alive.find(key.first);
    return it == alive.end() || it->second.starttime != key.second;
  };
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  for (auto it = g_confined.begin(); it != g_confined.end();) {
    it = is_dead(*it) ? g_confined.erase(it) : std::next(it);
  }
  for (auto it = g_backoff.begin(); it != g_backoff.end();) {
    it = is_dead(it->first) ? g_backoff.erase(it) : std::next(it);
  }
  return ++g_scan_seq;
}

// True when the thread is already confined or still backing off.
bool SkipThisScan(const ThreadKey &key, unsigned long long scan) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  if (g_confined.count(key) != 0) {
    return true;
  }
  const auto it = g_backoff.find(key);
  return it != g_backoff.end() && scan <= it->second.skip_through_scan;
}

void RecordNotProven(const ThreadKey &key, unsigned long long scan) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  Backoff &backoff = g_backoff[key];
  const unsigned long long shift = std::min<unsigned long long>(
      backoff.strikes, 4); // 1, 2, 4, 8, then capped.
  backoff.strikes += 1;
  backoff.skip_through_scan =
      scan + std::min<unsigned long long>(1ULL << shift, kMaxBackoffScans);
}

// --- Instruction-pointer probe ---------------------------------------------
// The handler only stores the interrupted PC of the targeted thread; all
// interpretation (dladdr) happens on the scanning thread. Every handler
// invocation counts itself in g_probe_inflight, and the scanner drains that
// count to zero after clearing g_probe_tid and before arming the next probe,
// so a handler that passed the tid check for thread T can never store into
// a later probe of thread U. All atomics are lock-free (async-signal-safe).
static_assert(std::atomic<pid_t>::is_always_lock_free,
              "probe tid slot must be lock-free");
static_assert(std::atomic<std::uintptr_t>::is_always_lock_free,
              "probe pc slot must be lock-free");
static_assert(std::atomic<int>::is_always_lock_free,
              "probe in-flight counter must be lock-free");
std::atomic<pid_t> g_probe_tid{0};
std::atomic<std::uintptr_t> g_probe_pc{0};
std::atomic<int> g_probe_inflight{0};

void ProbeHandler(int /*sig*/, siginfo_t * /*info*/, void *raw_context) {
  g_probe_inflight.fetch_add(1);
  if (CurrentThreadId() == g_probe_tid.load()) {
    const auto *context = static_cast<const ucontext_t *>(raw_context);
    std::uintptr_t pc = 0;
#if defined(__x86_64__)
    pc = static_cast<std::uintptr_t>(context->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
    pc = static_cast<std::uintptr_t>(context->uc_mcontext.pc);
#else
    (void)context;
#endif
    g_probe_pc.store(pc);
  }
  g_probe_inflight.fetch_sub(1);
}

std::mutex g_probe_mutex; // Serializes the single probe slot.

bool HandlerIsOurs(int sig) {
  struct sigaction current{};
  return sigaction(sig, nullptr, &current) == 0 &&
         (current.sa_flags & SA_SIGINFO) != 0 &&
         current.sa_sigaction == ProbeHandler;
}

// Returns the probe signal with our handler installed, or 0 when probing is
// unavailable. The handler is installed once and kept for the process
// lifetime: restoring SIG_DFL would let a late-delivered probe terminate the
// process (the default action for RT signals). Ownership is re-verified on
// every call, so a library that later claims the signal disables probing
// instead of receiving our probes. Caller holds g_probe_mutex.
int ProbeSignal() {
#if !defined(__x86_64__) && !defined(__aarch64__)
  return 0;
#else
  static bool install_attempted = false;
  const int sig = SIGRTMIN + kProbeSignalOffset;
  if (sig > SIGRTMAX) {
    return 0;
  }
  if (!install_attempted) {
    install_attempted = true;
    struct sigaction current{};
    if (sigaction(sig, nullptr, &current) == 0 &&
        (current.sa_flags & SA_SIGINFO) == 0 && current.sa_handler == SIG_DFL) {
      struct sigaction action{};
      action.sa_sigaction = ProbeHandler;
      action.sa_flags = SA_SIGINFO | SA_RESTART;
      sigemptyset(&action.sa_mask);
      (void)sigaction(sig, &action, nullptr);
    }
  }
  if (!HandlerIsOurs(sig)) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      log::Warn(kComponent, "probe signal SIGRTMIN+" +
                                std::to_string(kProbeSignalOffset) +
                                " is owned by another handler; spin tamer "
                                "disabled");
    }
    return 0;
  }
  return sig;
#endif
}

// Checked immediately before every probe signal:
//  - a thread that blocks or ignores the probe would queue RT signals (they
//    do not coalesce) or drop them;
//  - a thread that is not running/runnable ('R') is parked in a syscall
//    (epoll_wait, sem_timedwait, ...), where a signal could surface as EINTR
//    even with SA_RESTART. Spinners are always 'R', so nothing is lost.
bool SafeToProbe(pid_t tid, int sig) {
  if (ReadThreadState(tid) != 'R') {
    return false;
  }
  const unsigned long long bit = 1ULL << (sig - 1);
  unsigned long long masks[2] = {0, 0}; // SigBlk, SigIgn
  return ReadStatusFields(tid, {{"SigBlk:", 16}, {"SigIgn:", 16}}, masks) &&
         (masks[0] & bit) == 0 && (masks[1] & bit) == 0;
}

bool DrainHandlers() {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kHandlerDrainTimeoutMs);
  while (g_probe_inflight.load() != 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

enum class SampleResult { kPc, kNoReply, kSlotUnsafe };

// Interrupts `tid` and reports the PC it was executing. kNoReply: thread
// exited, is not safe to signal now, or did not run the handler in time.
// kSlotUnsafe: an in-flight handler did not finish, so the probe slot must
// not be reused this scan. Caller holds g_probe_mutex.
SampleResult SamplePc(pid_t pid, pid_t tid, int sig, std::uintptr_t *pc) {
  *pc = 0;
  if (!SafeToProbe(tid, sig)) {
    return SampleResult::kNoReply;
  }
  g_probe_pc.store(0);
  g_probe_tid.store(tid);
  if (syscall(SYS_tgkill, pid, tid, sig) == 0) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kPcReplyTimeoutMs);
    while ((*pc = g_probe_pc.load()) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  g_probe_tid.store(0);
  if (!DrainHandlers()) {
    return SampleResult::kSlotUnsafe;
  }
  return *pc != 0 ? SampleResult::kPc : SampleResult::kNoReply;
}

bool PcOwned(std::uintptr_t pc, const RocrSpinTamerOptions &options) {
  if (options.pc_owner) {
    return options.pc_owner(pc);
  }
  Dl_info info{};
  return dladdr(reinterpret_cast<void *>(pc), &info) != 0 &&
         info.dli_fname != nullptr &&
         std::strstr(info.dli_fname, options.code_owner.c_str()) != nullptr;
}

enum class Verdict { kOwner, kNotProven, kAbortScan };

// Samples `tid`'s PC until ownership is decided.
Verdict ProveOwner(pid_t pid, pid_t tid, const RocrSpinTamerOptions &options,
                   const std::atomic<bool> *cancelled) {
  std::lock_guard<std::mutex> lock(g_probe_mutex);
  const int sig = ProbeSignal();
  if (sig == 0) {
    return Verdict::kAbortScan;
  }
  int owned = 0;
  for (int i = 0; i < kPcSamples; ++i) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      return Verdict::kAbortScan;
    }
    std::uintptr_t pc = 0;
    const SampleResult result = SamplePc(pid, tid, sig, &pc);
    if (result == SampleResult::kSlotUnsafe) {
      return Verdict::kAbortScan;
    }
    if (result == SampleResult::kPc && PcOwned(pc, options)) {
      ++owned;
    }
    if (owned >= kPcOwnedRequired) {
      return Verdict::kOwner;
    }
    if (owned + (kPcSamples - 1 - i) < kPcOwnedRequired) {
      return Verdict::kNotProven; // Majority no longer reachable.
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return Verdict::kNotProven;
}

// Resolves the pin target. Returns false when pinning must be skipped
// (single-CPU mask). An invalid INFERFLUX_ROCR_TAME_CORE (unparseable, out
// of range, or outside the process affinity mask) warns and falls back to
// the highest allowed CPU.
bool ChooseTameCore(int *tame_core) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(getpid(), sizeof(allowed), &allowed) != 0) {
    log::Warn(kComponent,
              std::string("sched_getaffinity failed: ") + std::strerror(errno));
    return false;
  }
  int highest = -1;
  int count = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      highest = cpu;
      ++count;
    }
  }
  if (count <= 1) {
    return false; // Nothing to gain from pinning; demote only.
  }
  // Valid CPU indices are [0, CPU_SETSIZE - 1]; an index of CPU_SETSIZE
  // would make CPU_ISSET/CPU_SET address past the end of cpu_set_t.
  const int requested =
      ParseIntEnv("INFERFLUX_ROCR_TAME_CORE", -1, 0, CPU_SETSIZE - 1);
  const char *raw = std::getenv("INFERFLUX_ROCR_TAME_CORE");
  if (requested >= 0 && CPU_ISSET(requested, &allowed)) {
    *tame_core = requested;
    return true;
  }
  static std::atomic<bool> warned_invalid{false};
  if (raw != nullptr && *raw != '\0' && !warned_invalid.exchange(true)) {
    log::Warn(kComponent, std::string("INFERFLUX_ROCR_TAME_CORE=") + raw +
                              " is not a CPU in the allowed mask; using CPU " +
                              std::to_string(highest));
  }
  *tame_core = highest;
  return true;
}

std::atomic<int> g_next_spin_name{0}; // Unique rocr-spin-N across scans.

enum class Claim { kClaimed, kAlreadyConfined, kCapReached };

// Atomically claims `key` for confinement: concurrent scans (the background
// worker and a synchronous caller) can never confine or count it twice.
Claim ClaimConfinement(const ThreadKey &key) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  if (g_confined.count(key) != 0) {
    return Claim::kAlreadyConfined;
  }
  if (g_confined.size() >= kMaxConfinedAlive) {
    return Claim::kCapReached;
  }
  g_confined.insert(key);
  return Claim::kClaimed;
}

void UnclaimConfinement(const ThreadKey &key) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  g_confined.erase(key);
}

bool ConfineThread(const ThreadKey &key, int tame_core, bool pin) {
  const pid_t tid = key.first;
  // The proof can take a while; refuse if the TID now names another thread.
  ThreadSample now;
  if (!ReadThreadStat(tid, &now) || now.starttime != key.second) {
    return false;
  }
  switch (ClaimConfinement(key)) {
  case Claim::kAlreadyConfined:
    return false;
  case Claim::kCapReached: {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      log::Warn(kComponent, "cap of " + std::to_string(kMaxConfinedAlive) +
                                " confined live threads reached; confining no "
                                "more until some exit");
    }
    return false;
  }
  case Claim::kClaimed:
    break;
  }
  int demoted = 0;
  // PRIO_PROCESS with a tid targets that thread's nice value on Linux.
  if (setpriority(PRIO_PROCESS, tid, 19) == 0) {
    ++demoted;
  }
  sched_param idle{};
  if (sched_setscheduler(tid, SCHED_IDLE, &idle) == 0) {
    ++demoted;
  }
  if (demoted == 0) {
    UnclaimConfinement(key); // ESRCH (thread exited) or EPERM.
    return false;
  }
  if (pin) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(tame_core, &set);
    if (sched_setaffinity(tid, sizeof(set), &set) != 0) {
      log::Warn(kComponent, "affinity pin of tid " + std::to_string(tid) +
                                " failed: " + std::strerror(errno));
    }
  }
  // Best-effort rename for observability: same-tgid comm writes need no
  // capabilities, and "rocr-spin-N" fits the 15-char TASK_COMM_LEN limit.
  // No trailing newline: the kernel stores the written bytes verbatim.
  std::ofstream comm(TaskPath(tid, "comm"));
  if (comm) {
    comm << "rocr-spin-" << g_next_spin_name.fetch_add(1);
  }
  return true;
}

// Runnable for the whole window. With schedstat that is on-CPU + runqueue
// wait, which stays ~100% however contended the host is (a starved spinner
// waits rather than runs). schedstat that reports zeros for a thread that
// demonstrably ran (sched_info disabled) is inert, so it falls back to user
// time against the measured window; a loaded host can push that below the
// threshold, which is a miss, never a false confinement.
bool AlwaysRunnable(const ThreadSample &before, const ThreadSample &after,
                    long utime_delta, unsigned long long min_runnable_ns,
                    long min_busy_ticks) {
  if (before.has_schedstat && after.has_schedstat &&
      after.run_ns >= before.run_ns && after.wait_ns >= before.wait_ns) {
    const unsigned long long runnable =
        (after.run_ns - before.run_ns) + (after.wait_ns - before.wait_ns);
    if (runnable != 0) {
      return runnable >= min_runnable_ns;
    }
  }
  return utime_delta >= min_busy_ticks;
}

int TameRuntimeSpinThreadsImpl(const RocrSpinTamerOptions &options,
                               const std::atomic<bool> *cancelled) {
  if (!ParseBoolEnv("INFERFLUX_ROCR_TAME", true)) {
    return 0;
  }
  if (options.require_hsa_runtime && !HsaRuntimeLoaded()) {
    return 0;
  }
  if (!options.pc_owner && options.code_owner.empty()) {
    return 0; // An empty owner would match every object.
  }
  const pid_t pid = getpid();
  const pid_t self = CurrentThreadId();

  const auto first = SampleAllThreads();
  const auto window_start = std::chrono::steady_clock::now();
  for (int slept = 0; slept < kWindowMs; slept += kSleepSliceMs) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      return 0; // Process is exiting: don't touch metrics or statics.
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepSliceMs));
  }
  const auto second = SampleAllThreads();
  const unsigned long long scan = BeginScan(second);
  const long long window_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - window_start)
          .count();
  const unsigned long long min_runnable_ns =
      static_cast<unsigned long long>(window_ns) / 100 * kMinRunnablePercent;
  const long hz = sysconf(_SC_CLK_TCK);
  const long ticks_per_second = hz > 0 ? hz : 100; // No USER_HZ assumption.
  const long min_busy_ticks = static_cast<long>(
      window_ns / 1000000 * ticks_per_second * kMinRunnablePercent / 100000);

  // Stage 1 prefilter: threads that never blocked during the window.
  std::vector<ThreadKey> candidates;
  for (const auto &entry : second) {
    const pid_t tid = entry.first;
    if (tid == pid || tid == self) {
      continue;
    }
    const auto it_before = first.find(tid);
    if (it_before == first.end()) {
      continue; // Appeared mid-window; the next scan sees it whole.
    }
    const ThreadSample &before = it_before->second;
    const ThreadSample &after = entry.second;
    if (after.starttime != before.starttime ||
        after.utime_ticks < before.utime_ticks ||
        after.stime_ticks < before.stime_ticks ||
        after.voluntary_switches < before.voluntary_switches) {
      continue; // TID reused by a different thread inside the window.
    }
    const ThreadKey key{tid, after.starttime};
    if (SkipThisScan(key, scan)) {
      continue;
    }
    const long utime_delta = after.utime_ticks - before.utime_ticks;
    const long stime_delta = after.stime_ticks - before.stime_ticks;
    const long long voluntary_delta =
        after.voluntary_switches - before.voluntary_switches;
    if (stime_delta <= kMaxStimeDeltaTicks && voluntary_delta == 0 &&
        AlwaysRunnable(before, after, utime_delta, min_runnable_ns,
                       min_busy_ticks)) {
      candidates.push_back(key);
    }
  }
  if (candidates.empty()) {
    return 0;
  }

  // Stage 2 proof: only threads provably executing the code owner's code.
  // The pin target is resolved lazily, on the first proven thread.
  bool core_resolved = false;
  bool pin = false;
  int tame_core = -1;
  int confined = 0;
  for (const ThreadKey &key : candidates) {
    const Verdict verdict = ProveOwner(pid, key.first, options, cancelled);
    if (verdict == Verdict::kAbortScan) {
      break;
    }
    if (verdict == Verdict::kNotProven) {
      RecordNotProven(key, scan);
      continue;
    }
    if (!core_resolved) {
      pin = ChooseTameCore(&tame_core);
      core_resolved = true;
    }
    if (ConfineThread(key, tame_core, pin)) {
      ++confined;
    }
  }

  if (confined > 0) {
    GlobalMetrics().RecordRuntimeSpinThreadsConfined(confined);
    log::Info(
        kComponent,
        "confined " + std::to_string(confined) +
            " busy-spin runtime thread(s) (nice 19, SCHED_IDLE" +
            (pin ? std::string(", pinned to CPU ") + std::to_string(tame_core)
                 : std::string(", pin skipped")) +
            ")");
  } else {
    log::Debug(kComponent, std::to_string(candidates.size()) +
                               " never-blocking thread(s) found; none proven "
                               "to run in " +
                               (options.pc_owner ? std::string("the test owner")
                                                 : options.code_owner));
  }
  return confined;
}

// Background scanner with a fixed schedule: immediately, +10 s, +60 s, then
// every 5 min. Blocks on a condition variable between scans (never spins).
// Never destroyed (see Instance()).
class TamerWorker {
public:
  // Leaked on purpose so it outlives static destruction. The logger and
  // metrics registry are namespace-scope globals constructed before main, so
  // this atexit hook (registered later, on first use) runs before they are
  // destroyed; it stops and joins the worker.
  static TamerWorker &Instance() {
    static TamerWorker *instance = [] {
      auto *worker = new TamerWorker();
      std::atexit([] { Instance().Shutdown(); });
      return worker;
    }();
    return *instance;
  }

  void Start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_ || thread_.joinable()) {
      return; // Idempotent.
    }
    thread_ = std::thread(&TamerWorker::Loop, this);
  }

private:
  TamerWorker() = default;

  void Shutdown() {
    std::thread worker;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      worker = std::move(thread_);
    }
    // Cuts an in-flight scan short (<= one 50 ms slice or one PC probe).
    cancelled_.store(true);
    cv_.notify_all();
    if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) {
      worker.join();
    } else if (worker.joinable()) {
      worker.detach(); // exit() called from the worker itself.
    }
  }

  void Loop() {
    SetCurrentThreadName("if-rocr-tamer");
    using namespace std::chrono_literals;
    const std::chrono::steady_clock::duration kDelays[] = {0s, 10s, 60s};
    constexpr auto kPeriod = 5min;
    std::size_t next_delay = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      const auto delay = next_delay < std::size(kDelays)
                             ? kDelays[next_delay++]
                             : std::chrono::steady_clock::duration(kPeriod);
      if (cv_.wait_for(lock, delay, [this] { return stop_; })) {
        return;
      }
      lock.unlock();
      try {
        TameRuntimeSpinThreadsImpl(RocrSpinTamerOptions{}, &cancelled_);
      } catch (...) {
        // Best-effort hygiene must never take the server down.
      }
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::atomic<bool> cancelled_{false};
  std::thread thread_;
};
#endif // defined(__linux__)

} // namespace

int TameRuntimeSpinThreads(const RocrSpinTamerOptions &options) {
#if defined(__linux__)
  return TameRuntimeSpinThreadsImpl(options, nullptr);
#else
  (void)options;
  return 0;
#endif
}

void StartRuntimeSpinTamer() noexcept {
#if defined(__linux__)
  try {
    if (!ParseBoolEnv("INFERFLUX_ROCR_TAME", true)) {
      return;
    }
    // libhsa-runtime64 is linked into ROCm builds, so it is mapped from exec
    // whenever HIP could ever initialize; other builds get no thread at all.
    if (!HsaRuntimeLoaded()) {
      return;
    }
    TamerWorker::Instance().Start();
  } catch (const std::exception &ex) {
    try {
      log::Warn(kComponent,
                std::string("spin tamer failed to start: ") + ex.what());
    } catch (...) {
    }
  } catch (...) {
  }
#endif
}

} // namespace inferflux
