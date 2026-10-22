#include "server/rocr_spin_tamer.h"

#include "runtime/string_utils.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#endif

namespace inferflux {
namespace {

#if defined(__linux__)
const char *kComponent = "rocr_tamer";

constexpr int kWindowMs = 1200;
constexpr int kSleepSliceMs = 50;
constexpr int kMinBusyPercent = 80;
constexpr long kMaxStimeDeltaTicks = 2;
constexpr int kMaxConfined = 4; // ROCm spawns 2; cap guards against storms.

struct ThreadSample {
  long utime_ticks = 0;
  long stime_ticks = 0;
  long long voluntary_switches = 0;
};

pid_t CurrentTid() { return static_cast<pid_t>(syscall(SYS_gettid)); }

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

// comm may contain spaces/parens, so parse fields after the LAST ')'.
// Token layout after ')': tok[0]=state(3) ... tok[11]=utime(14),
// tok[12]=stime(15) of the /proc stat field numbering.
bool ReadThreadStat(pid_t tid, std::string *comm, ThreadSample *out) {
  std::ifstream file("/proc/self/task/" + std::to_string(tid) + "/stat");
  std::string line;
  if (!file || !std::getline(file, line)) {
    return false; // TID vanished mid-sample; caller skips it.
  }
  const std::size_t open = line.find('(');
  const std::size_t close = line.rfind(')');
  if (open == std::string::npos || close == std::string::npos || close < open) {
    return false;
  }
  if (comm) {
    *comm = line.substr(open + 1, close - open - 1);
  }
  std::istringstream fields(line.substr(close + 1));
  std::vector<std::string> tokens;
  std::string token;
  while (fields >> token) {
    tokens.push_back(token);
  }
  if (tokens.size() < 13) {
    return false;
  }
  out->utime_ticks = std::strtol(tokens[11].c_str(), nullptr, 10);
  out->stime_ticks = std::strtol(tokens[12].c_str(), nullptr, 10);
  return true;
}

bool ReadVoluntarySwitches(pid_t tid, long long *out) {
  std::ifstream file("/proc/self/task/" + std::to_string(tid) + "/status");
  std::string line;
  while (file && std::getline(file, line)) {
    if (line.rfind("voluntary_ctxt_switches:", 0) == 0) {
      *out = std::strtoll(line.c_str() + 25, nullptr, 10);
      return true;
    }
  }
  return false; // TID vanished; caller skips it.
}

// Snapshots utime/stime/voluntary counters per thread. Thread comms are
// read on the first snapshot only: classification keys off the creation
// comm, and a confined thread may already have been renamed by a prior run.
std::unordered_map<pid_t, ThreadSample>
SampleAllThreads(std::unordered_map<pid_t, std::string> *comms) {
  std::unordered_map<pid_t, ThreadSample> samples;
  for (const pid_t tid : ListThreads()) {
    ThreadSample sample;
    std::string comm;
    if (!ReadThreadStat(tid, comms ? &comm : nullptr, &sample)) {
      continue;
    }
    if (!ReadVoluntarySwitches(tid, &sample.voluntary_switches)) {
      continue;
    }
    if (comms) {
      comms->emplace(tid, std::move(comm));
    }
    samples.emplace(tid, sample);
  }
  return samples;
}

// Resolves the pin target. Returns false when pinning must be skipped
// (single-CPU mask). An explicitly requested core outside the process
// affinity mask warns once and falls back to the highest allowed CPU.
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
  int requested = ParseIntEnv("INFERFLUX_ROCR_TAME_CORE", -1, 0, CPU_SETSIZE);
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

bool ConfineThread(pid_t tid, int index, int tame_core, bool pin) {
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
  std::ofstream comm("/proc/self/task/" + std::to_string(tid) + "/comm");
  if (comm) {
    comm << "rocr-spin-" << index << "\n";
  }
  return true;
}

int TameRuntimeSpinThreadsImpl(const std::atomic<bool> *cancelled) {
  if (!ParseBoolEnv("INFERFLUX_ROCR_TAME", true)) {
    return 0;
  }
  const pid_t pid = getpid();
  const pid_t self = CurrentTid();

  std::ifstream comm_file("/proc/self/comm");
  std::string process_comm;
  std::getline(comm_file, process_comm);
  if (!process_comm.empty() && process_comm.back() == '\n') {
    process_comm.pop_back();
  }
  if (process_comm.empty()) {
    return 0; // No comm baseline to discriminate against; do nothing.
  }

  std::unordered_map<pid_t, std::string> comms;
  const auto first = SampleAllThreads(&comms);

  for (int slept = 0; slept < kWindowMs; slept += kSleepSliceMs) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      return 0; // Process is exiting: don't touch metrics or statics.
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepSliceMs));
  }

  const long hz = sysconf(_SC_CLK_TCK);
  const long ticks_per_second = hz > 0 ? hz : 100; // No USER_HZ assumption.
  const long min_busy_ticks =
      static_cast<long>(static_cast<long long>(kWindowMs) * ticks_per_second *
                        kMinBusyPercent / 100000);

  int tame_core = -1;
  const bool pin = ChooseTameCore(&tame_core);

  const auto second = SampleAllThreads(nullptr);
  int confined = 0;
  for (const auto &entry : second) {
    if (confined >= kMaxConfined) {
      break;
    }
    const pid_t tid = entry.first;
    if (tid == pid || tid == self) {
      continue;
    }
    const auto it_before = first.find(tid);
    if (it_before == first.end()) {
      continue; // Appeared mid-window: not a hipInit-era poller.
    }
    const auto it_comm = comms.find(tid);
    if (it_comm == comms.end() || it_comm->second != process_comm) {
      continue; // Named thread: not an unnamed runtime poller.
    }
    const ThreadSample &before = it_before->second;
    const ThreadSample &after = entry.second;
    const long utime_delta = after.utime_ticks - before.utime_ticks;
    const long stime_delta = after.stime_ticks - before.stime_ticks;
    const long long voluntary_delta =
        after.voluntary_switches - before.voluntary_switches;
    // Pure userspace spin: near-100% user time, no kernel time, and no
    // voluntary sleeps. Preemption only bumps *non*voluntary switches, so
    // this discriminator holds even on a fully loaded machine.
    if (utime_delta >= min_busy_ticks && stime_delta <= kMaxStimeDeltaTicks &&
        voluntary_delta == 0) {
      if (ConfineThread(tid, confined, tame_core, pin)) {
        ++confined;
      }
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
#endif // defined(__linux__)

#if defined(__linux__)
void RunTamerWorker(std::atomic<bool> *cancelled) {
  TameRuntimeSpinThreadsImpl(cancelled);
}
#endif

} // namespace

int TameRuntimeSpinThreads() {
#if defined(__linux__)
  return TameRuntimeSpinThreadsImpl(nullptr);
#else
  return 0;
#endif
}

void TameRuntimeSpinThreadsAsync() {
#if defined(__linux__)
  struct TamerWorker {
    std::thread thread;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> started{false};

    ~TamerWorker() {
      // Runs at exit in reverse construction order: g_metrics and the
      // logger are constructed before this worker (first call is inside
      // main), so they are still alive while we join.
      cancelled.store(true);
      if (thread.joinable()) {
        thread.join();
      }
    }
  };
  static TamerWorker worker;
  if (worker.started.exchange(true)) {
    return; // Idempotent across re-entry.
  }
  try {
    worker.thread = std::thread(RunTamerWorker, &worker.cancelled);
  } catch (const std::exception &ex) {
    worker.started.store(false);
    log::Warn(kComponent,
              std::string("spin-tamer worker spawn failed: ") + ex.what());
  }
#endif
}

} // namespace inferflux
