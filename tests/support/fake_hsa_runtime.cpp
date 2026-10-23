// Stand-in for libhsa-runtime64, built as a dlopen-only module whose
// basename starts with "libhsa-runtime64" (see CMakeLists.txt). Threads it
// starts have their start routine inside this object, exactly like ROCr's,
// so the pthread_create interposer must recognize and tame them.

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <ctime>
#include <memory>

extern "C" {
struct FakeHsaReport {
  int cpu_count;
  int only_cpu; // Lowest CPU in the thread's mask.
  int nice;
  char name[16];
  int tls_dtor_ran;
  int done; // Set last (atomically) by detached spawns.
};
}

namespace {

enum Mode { kReport = 0, kPthreadExit = 1, kCancellable = 2, kThreadLocal = 3 };

struct TlsProbe {
  FakeHsaReport *report = nullptr;
  ~TlsProbe() {
    if (report != nullptr) {
      report->tls_dtor_ran = 1;
    }
  }
};
thread_local TlsProbe g_tls_probe;

struct Job {
  FakeHsaReport *report;
  int mode;
};

void Fill(FakeHsaReport *report) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  report->cpu_count = 0;
  report->only_cpu = -1;
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    report->cpu_count = CPU_COUNT(&mask);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &mask)) {
        report->only_cpu = cpu;
        break;
      }
    }
  }
  report->nice =
      getpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)));
  report->name[0] = '\0';
  (void)pthread_getname_np(pthread_self(), report->name, sizeof(report->name));
}

// File-static (hidden) like ROCr's internal thread start routine; dladdr
// still resolves the containing object from its address.
void *FakeHsaMain(void *raw_job) {
  const std::unique_ptr<Job> job(static_cast<Job *>(raw_job));
  FakeHsaReport *const report = job->report;
  switch (job->mode) {
  case kPthreadExit:
    Fill(report);
    pthread_exit(report);
  case kCancellable:
    for (;;) {
      pthread_testcancel();
      const timespec pause{0, 1000000};
      nanosleep(&pause, nullptr);
    }
  case kThreadLocal:
    g_tls_probe.report = report;
    Fill(report);
    return report;
  case kReport:
  default:
    Fill(report);
    __atomic_store_n(&report->done, 1, __ATOMIC_RELEASE);
    return report;
  }
}

} // namespace

extern "C" {

// Starts a thread the way ROCr does. `widen_attr` sets an explicit
// all-online-CPU affinity through the attribute, like ROCr's
// pthread_attr_setaffinity_np. Returns pthread_create's result.
__attribute__((visibility("default"))) int
fake_hsa_spawn(pthread_t *out, FakeHsaReport *report, int mode, int widen_attr,
               int detached) {
  std::memset(report, 0, sizeof(*report));
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  if (widen_attr) {
    cpu_set_t all;
    CPU_ZERO(&all);
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    for (long cpu = 0; cpu < online && cpu < CPU_SETSIZE; ++cpu) {
      CPU_SET(static_cast<int>(cpu), &all);
    }
    pthread_attr_setaffinity_np(&attr, sizeof(all), &all);
  }
  if (detached) {
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  }
  auto job = std::make_unique<Job>(Job{report, mode});
  const int rc = pthread_create(out, &attr, &FakeHsaMain, job.get());
  if (rc == 0) {
    (void)job.release(); // Owned by the new thread now.
  }
  pthread_attr_destroy(&attr);
  return rc;
}

// Starts a report thread with a caller-provided attribute (error paths).
__attribute__((visibility("default"))) int
fake_hsa_spawn_with_attr(pthread_t *out, const pthread_attr_t *attr,
                         FakeHsaReport *report) {
  std::memset(report, 0, sizeof(*report));
  auto job = std::make_unique<Job>(Job{report, kReport});
  const int rc = pthread_create(out, attr, &FakeHsaMain, job.get());
  if (rc == 0) {
    (void)job.release();
  }
  return rc;
}

// An address inside this object (for dladdr-based checks).
__attribute__((visibility("default"))) void *fake_hsa_symbol() {
  return reinterpret_cast<void *>(&FakeHsaMain);
}

} // extern "C"
