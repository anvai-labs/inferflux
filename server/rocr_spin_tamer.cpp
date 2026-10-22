#include "server/rocr_spin_tamer.h"

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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
// PC proof: a strict majority of samples must land in the code owner; the
// rest may be vDSO/libc helpers the spin loop calls without a syscall.
constexpr int kPcSamples = 5;
constexpr int kPcOwnedRequired = 3;
// Generous: a runnable-but-starved spinner on a loaded host may wait on a
// runqueue before the handler runs. Scans are background work.
constexpr int kPcReplyTimeoutMs = 500;
// The ROCr runtime spawns 2 pollers; this only bounds work and log noise.
constexpr int kMaxConfined = 8;
// Offset from SIGRTMIN for the PC probe. glibc reserves the RT signals below
// its SIGRTMIN; nothing in InferFlux or llama.cpp/ggml uses RT signals, and
// a foreign handler (at install time or any later scan) disables probing.
constexpr int kProbeSignalOffset = 5;

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

pid_t CurrentTid() { return static_cast<pid_t>(syscall(SYS_gettid)); }

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

bool ReadVoluntarySwitches(pid_t tid, long long *out) {
  std::ifstream file(TaskPath(tid, "status"));
  std::string line;
  while (file && std::getline(file, line)) {
    if (line.rfind("voluntary_ctxt_switches:", 0) == 0) {
      // strtoll skips the separator whitespace itself, so this does not
      // depend on the exact /proc padding after the 24-char key.
      *out = std::strtoll(line.c_str() + 24, nullptr, 10);
      return true;
    }
  }
  return false; // TID vanished; caller skips it.
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
    if (!ReadThreadStat(tid, &sample) ||
        !ReadVoluntarySwitches(tid, &sample.voluntary_switches)) {
      continue;
    }
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

// Threads already confined, by (tid, starttime): never re-confined or double
// counted, and immune to TID reuse.
std::mutex g_confined_mutex;
std::set<std::pair<pid_t, unsigned long long>> g_confined;

bool AlreadyConfined(pid_t tid, unsigned long long starttime) {
  std::lock_guard<std::mutex> lock(g_confined_mutex);
  return g_confined.count({tid, starttime}) != 0;
}

// --- Instruction-pointer probe ---------------------------------------------
// The handler only stores the interrupted PC of the targeted thread; all
// interpretation (dladdr) happens on the scanning thread. Both atomics are
// lock-free, so the handler is async-signal-safe.
static_assert(std::atomic<pid_t>::is_always_lock_free,
              "probe tid slot must be lock-free");
static_assert(std::atomic<std::uintptr_t>::is_always_lock_free,
              "probe pc slot must be lock-free");
std::atomic<pid_t> g_probe_tid{0};
std::atomic<std::uintptr_t> g_probe_pc{0};

void ProbeHandler(int /*sig*/, siginfo_t * /*info*/, void *raw_context) {
  if (static_cast<pid_t>(syscall(SYS_gettid)) !=
      g_probe_tid.load(std::memory_order_acquire)) {
    return; // Late delivery for an abandoned probe: ignore.
  }
  const auto *context = static_cast<const ucontext_t *>(raw_context);
  std::uintptr_t pc = 0;
#if defined(__x86_64__)
  pc = static_cast<std::uintptr_t>(context->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
  pc = static_cast<std::uintptr_t>(context->uc_mcontext.pc);
#else
  (void)context;
#endif
  g_probe_pc.store(pc, std::memory_order_release);
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

// Interrupts `tid` and returns the PC it was executing, or 0 on no reply
// (signal blocked, thread exited, or timeout). Caller holds g_probe_mutex.
std::uintptr_t SamplePc(pid_t pid, pid_t tid, int sig) {
  g_probe_pc.store(0, std::memory_order_release);
  g_probe_tid.store(tid, std::memory_order_release);
  std::uintptr_t pc = 0;
  if (syscall(SYS_tgkill, pid, tid, sig) == 0) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kPcReplyTimeoutMs);
    while ((pc = g_probe_pc.load(std::memory_order_acquire)) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  g_probe_tid.store(0, std::memory_order_release);
  return pc;
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

// True when a majority of sampled PCs of `tid` belong to the code owner.
bool ProvenOwnerThread(pid_t pid, pid_t tid,
                       const RocrSpinTamerOptions &options,
                       const std::atomic<bool> *cancelled) {
  std::lock_guard<std::mutex> lock(g_probe_mutex);
  const int sig = ProbeSignal();
  if (sig == 0) {
    return false;
  }
  int owned = 0;
  for (int i = 0; i < kPcSamples; ++i) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      return false;
    }
    const std::uintptr_t pc = SamplePc(pid, tid, sig);
    if (pc != 0 && PcOwned(pc, options)) {
      ++owned;
    }
    if (owned >= kPcOwnedRequired) {
      return true;
    }
    if (owned + (kPcSamples - 1 - i) < kPcOwnedRequired) {
      return false; // Majority no longer reachable.
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

// Resolves the pin target. Returns false when pinning must be skipped
// (single-CPU mask). An explicitly requested core outside the process
// affinity mask warns and falls back to the highest allowed CPU.
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
  if (requested >= 0) {
    if (CPU_ISSET(requested, &allowed)) {
      *tame_core = requested;
      return true;
    }
    log::Warn(kComponent,
              "INFERFLUX_ROCR_TAME_CORE=" + std::to_string(requested) +
                  " is outside the allowed CPU mask; using CPU " +
                  std::to_string(highest));
  }
  *tame_core = highest;
  return true;
}

std::atomic<int> g_next_spin_name{0}; // Unique rocr-spin-N across scans.

bool ConfineThread(pid_t tid, unsigned long long starttime, int tame_core,
                   bool pin) {
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
    return false; // ESRCH (thread exited) or EPERM: give up on this one.
  }
  {
    std::lock_guard<std::mutex> lock(g_confined_mutex);
    g_confined.insert({tid, starttime});
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
  std::ofstream comm(TaskPath(tid, "comm"));
  if (comm) {
    comm << "rocr-spin-" << g_next_spin_name.fetch_add(1) << "\n";
  }
  return true;
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
  const pid_t self = CurrentTid();

  const auto first = SampleAllThreads();
  const auto window_start = std::chrono::steady_clock::now();
  for (int slept = 0; slept < kWindowMs; slept += kSleepSliceMs) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      return 0; // Process is exiting: don't touch metrics or statics.
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepSliceMs));
  }
  const auto second = SampleAllThreads();
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
  std::vector<std::pair<pid_t, unsigned long long>> candidates;
  for (const auto &entry : second) {
    const pid_t tid = entry.first;
    if (tid == pid || tid == self) {
      continue;
    }
    const auto it_before = first.find(tid);
    if (it_before == first.end()) {
      continue; // Appeared mid-window; a later scan will see it whole.
    }
    const ThreadSample &before = it_before->second;
    const ThreadSample &after = entry.second;
    if (after.starttime != before.starttime ||
        after.utime_ticks < before.utime_ticks ||
        after.stime_ticks < before.stime_ticks ||
        after.voluntary_switches < before.voluntary_switches) {
      continue; // TID reused by a different thread inside the window.
    }
    if (AlreadyConfined(tid, after.starttime)) {
      continue;
    }
    const long utime_delta = after.utime_ticks - before.utime_ticks;
    const long stime_delta = after.stime_ticks - before.stime_ticks;
    const long long voluntary_delta =
        after.voluntary_switches - before.voluntary_switches;
    // Runnable for the whole window: with schedstat that is on-CPU +
    // runqueue-wait time, which stays ~100% however contended the host is
    // (a starved spinner waits rather than runs). Without schedstat, user
    // time against the measured window; a loaded host can push that below
    // the threshold, which is a miss, never a false confinement.
    bool always_runnable = false;
    if (before.has_schedstat && after.has_schedstat) {
      always_runnable =
          after.run_ns >= before.run_ns && after.wait_ns >= before.wait_ns &&
          (after.run_ns - before.run_ns) + (after.wait_ns - before.wait_ns) >=
              min_runnable_ns;
    } else {
      always_runnable = utime_delta >= min_busy_ticks;
    }
    if (always_runnable && utime_delta > 0 &&
        stime_delta <= kMaxStimeDeltaTicks && voluntary_delta == 0) {
      candidates.emplace_back(tid, after.starttime);
    }
  }
  if (candidates.empty()) {
    return 0;
  }

  // Stage 2 proof: only threads provably executing the code owner's code.
  int tame_core = -1;
  const bool pin = ChooseTameCore(&tame_core);
  int confined = 0;
  for (const auto &[tid, starttime] : candidates) {
    if (confined >= kMaxConfined ||
        (cancelled && cancelled->load(std::memory_order_relaxed))) {
      break;
    }
    if (ProvenOwnerThread(pid, tid, options, cancelled) &&
        ConfineThread(tid, starttime, tame_core, pin)) {
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
  }
  return confined;
}

// Background scanner. Requests set `pending_` and wake the worker; requests
// landing mid-scan leave `pending_` set, so they coalesce into exactly one
// follow-up scan. Between scans the worker blocks on the condition variable
// (never spins). Never destroyed (see Instance()).
class TamerWorker {
public:
  // Leaked on purpose: Request() may run on any thread (HTTP workers, the
  // registry watcher) while the process exits, so the object must outlive
  // static destruction. The atexit hook is registered after GlobalMetrics()
  // and the logger exist (first use is inside main), so it runs before they
  // are torn down: it stops and joins the worker, and later Request() calls
  // see stop_ and return.
  static TamerWorker &Instance() {
    static TamerWorker *instance = [] {
      auto *worker = new TamerWorker();
      std::atexit([] { Instance().Shutdown(); });
      return worker;
    }();
    return *instance;
  }

  void Request() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_) {
      return;
    }
    pending_ = true;
    if (!thread_.joinable()) {
      try {
        thread_ = std::thread(&TamerWorker::Loop, this);
      } catch (const std::exception &ex) {
        pending_ = false;
        log::Warn(kComponent,
                  std::string("spin-tamer worker spawn failed: ") + ex.what());
        return;
      }
    }
    cv_.notify_one();
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
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      cv_.wait(lock, [this] { return stop_ || pending_; });
      if (stop_) {
        return;
      }
      pending_ = false;
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
  bool pending_ = false;
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

void TameRuntimeSpinThreadsAsync() {
#if defined(__linux__)
  TamerWorker::Instance().Request();
#endif
}

} // namespace inferflux
