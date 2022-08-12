/* 
    ==============================================================================================
     OpenCL_Interface class source file (William Denny, 12th Aug 2022)
    ==============================================================================================
*/

#include "OpenCL_Interface.hpp"

// empty constructor
OpenCL_Interface::OpenCL_Interface() 
{
    std::cout << "Empty constructor called" << std::endl;
}

// main constructor
OpenCL_Interface::OpenCL_Interface(
    const int& platformIndex, 
    const int& deviceIndex
)
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    // get vector of platforms
    cl::Platform::get(&platforms);

    // abort if no platforms found
    assert(("No platforms found", platforms.size() > 0));

    // get target platform
    platform = platforms[platformIndex];

    // get vector of devices
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // abort if no devices found
    assert(("No devices found", devices.size() > 0));

    // get target device
    device = devices[deviceIndex];

    // Output success to user
    std::cout << "Connected to accelerator device: ";
    std::cout << device.getInfo<CL_DEVICE_VENDOR>() << ", " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;

    // create context
    context = cl::Context({ device });

    // create queue
    queue = cl::CommandQueue(context, device);
}

// add kernel
void OpenCL_Interface::addKernel(
    const std::string& kernelPath, 
    const std::string& kernelName,
    const std::vector<int>& inputArraySizes,
    const std::vector<int>& outputArraySizes
)
{
    // create kernel interface object
    /* requires:
        - device and context pointers
        - kernel path and name
        - intput and output array sizes
    */
    OpenCL_KernelInterface kernelInterface = OpenCL_KernelInterface(&device, &context, kernelPath, kernelName, inputArraySizes, outputArraySizes);

    // add kernel interface to vector
    kernelInterfaces.push_back(kernelInterface);
}

// run kernel
void OpenCL_Interface::runKernel(
    const int& index,
    const int& numWorkitems,
    const int& workgroupSize,
    const std::vector<float*>& inputArrayPtrs, 
    const std::vector<float*>& outputArrayPtrs
)
{
    // add input to input buffer
    kernelInterfaces[index].setInputBuffers(&queue, inputArrayPtrs);

    // set kernel arguments
    kernelInterfaces[index].setKernelArgs();

    // run kernel
    kernelInterfaces[index].runKernel(&queue, numWorkitems, workgroupSize);

    // wait for kernel to finish
    queue.finish();

    // transfer output buffer to output
    kernelInterfaces[index].getOutputBuffers(&queue, outputArrayPtrs);
}