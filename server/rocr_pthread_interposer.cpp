// Defines pthread_create for the executable it is linked into, so the ROCr
// thread tamer can recognize threads started by libhsa-runtime64 at creation
// (see runtime/rocr_spin_tamer.h). Linked ONLY into ROCm-enabled
// (ENABLE_ROCM) inferfluxd builds and rocr_interposer_tests -- never into
// inferflux_core, which inferctl, the benchmarks and the other tests also link.
// The symbol is exported with cmake/rocr_pthread_interposer.dynlist so
// libraries (libhsa-runtime64, loaded as a dependency or dlopen'd RTLD_LOCAL
// via ggml-hip) bind to it.

#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) ||     \
    __has_feature(memory_sanitizer) || __has_feature(hwaddress_sanitizer)
#define INFERFLUX_NO_PTHREAD_INTERPOSER 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) ||           \
    defined(__SANITIZE_HWADDRESS__)
#define INFERFLUX_NO_PTHREAD_INTERPOSER 1
#endif

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__)) &&     \
    !defined(INFERFLUX_NO_PTHREAD_INTERPOSER)

#include "runtime/rocr_spin_tamer.h"

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>

namespace {

using PthreadCreateFn = int (*)(pthread_t *, const pthread_attr_t *,
                                void *(*)(void *), void *);

// Constant-initialized: library constructors can create threads before this
// file's dynamic initializers run.
std::atomic<PthreadCreateFn> g_real_pthread_create{nullptr};

PthreadCreateFn RealPthreadCreate() noexcept {
  PthreadCreateFn fn = g_real_pthread_create.load(std::memory_order_acquire);
  if (fn != nullptr) {
    return fn;
  }
  // dlsym never creates threads, so this cannot recurse. It returns the C
  // library's default pthread_create version.
  fn = reinterpret_cast<PthreadCreateFn>(dlsym(RTLD_NEXT, "pthread_create"));
  if (fn == nullptr || fn == &pthread_create) {
    static const char kMessage[] =
        "inferflux: cannot resolve the C library's pthread_create\n";
    (void)!write(2, kMessage, sizeof(kMessage) - 1);
    std::abort(); // A build/packaging defect; never limp on without threads.
  }
  g_real_pthread_create.store(fn, std::memory_order_release);
  return fn;
}

struct TameBox {
  void *(*start_routine)(void *);
  void *arg;
  int core; // >= 0: tame this ROCr thread onto `core`.
  // inherited.creator_cpu >= 0: created by a tamed thread; what to undo.
  inferflux::InheritedState inherited;
};

// Deliberately NOT noexcept, with no try/catch and no RAII objects:
// pthread_exit() and cancellation in the ROCr routine unwind through this
// frame as a forced unwind, which a noexcept frame would turn into
// std::terminate and a catch(...) would abort on. The box is freed before
// the real routine runs, so nothing is left to clean up on those paths.
void *RocrTameTrampoline(void *raw_box) {
  const TameBox box = *static_cast<TameBox *>(raw_box);
  std::free(raw_box);
  if (box.core >= 0) {
    inferflux::TameCurrentRocrThread(box.core, box.inherited);
  } else {
    inferflux::RestoreInheritedScheduling(box.inherited);
  }
  return box.start_routine(box.arg);
}

// Resolves the real function off the hot path and registers this
// interposer for the startup self-check. Creations that happen before this
// runs (library constructors) resolve lazily.
[[maybe_unused]] const bool g_interposer_ready = [] {
  (void)RealPthreadCreate();
  inferflux::NoteRocrPthreadInterposerInstalled(
      reinterpret_cast<void *>(&pthread_create));
  return true;
}();

} // namespace

// glibc declares pthread_create noexcept in C++; the definition must match.
extern "C" __attribute__((visibility("default"), used)) int
pthread_create(pthread_t *__restrict thread,
               const pthread_attr_t *__restrict attr,
               void *(*start_routine)(void *), void *__restrict arg) noexcept {
  // Saved first: the lazy dlsym, dladdr, getenv and malloc must not leak
  // errno to the caller.
  const int saved_errno = errno;
  const PthreadCreateFn real = RealPthreadCreate();
  const int core = inferflux::RocrTameCoreForNewThread(start_routine);
  // A tamed thread's children inherit its single-CPU mask and name: a
  // non-ROCr child gets them undone in the trampoline, a ROCr child is
  // tamed against what it would have inherited without the tamer.
  inferflux::InheritedState inherited;
  inherited.creator_cpu = inferflux::InheritedPinToUndo(attr, &inherited);
  if (core >= 0 || inherited.creator_cpu >= 0) {
    if (auto *box = static_cast<TameBox *>(std::malloc(sizeof(TameBox)))) {
      *box = TameBox{start_routine, arg, core, inherited};
      errno = saved_errno;
      const int rc = real(thread, attr, &RocrTameTrampoline, box);
      if (rc != 0) {
        const int create_errno = errno;
        std::free(box); // The child never ran, so the box is still ours.
        errno = create_errno;
      }
      return rc;
    }
    // Out of memory for the box: create the thread untouched.
  }
  errno = saved_errno;
  return real(thread, attr, start_routine, arg);
}

#define INFERFLUX_PTHREAD_INTERPOSER_COMPILED 1
#endif

// Lets tests skip (rather than fail) where the interposer is compiled out.
extern "C" __attribute__((visibility("default")))
const int inferflux_rocr_pthread_interposer_compiled =
#if defined(INFERFLUX_PTHREAD_INTERPOSER_COMPILED)
    1;
#else
    0;
#endif
