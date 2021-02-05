#ifndef WEFT_BACKEND_SCHEDULER_H
#define WEFT_BACKEND_SCHEDULER_H

#include <vector>

#include "device.h"
#include "kernel.h"

namespace weft {

class Scheduler {
 public:
  Scheduler();

  void schedule(const kernel::Function &func,
                const kernel::ExecutionArgs &execution);

 private:
  int device_count_;
  std::vector<Device> devices_;

  int CuInitialize();
};

}  // namespace weft

#endif  // WEFT_BACKEND_SCHEDULER_H
