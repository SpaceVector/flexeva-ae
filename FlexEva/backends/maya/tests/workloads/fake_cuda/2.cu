#include <stdio.h>
#include <nvml.h>

int main() {
    nvmlReturn_t result;
    unsigned int deviceCount = 0;

    // 2. 显式调用 nvmlDeviceGetCount_v2
    result = nvmlDeviceGetCount_v2(&deviceCount);
    printf("nvmlDeviceGetCount_v2 返回值: %d\n", result);

    return 0;
}
