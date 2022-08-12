#include <CL/opencl.hpp>

#include "OpenCLInterface.hpp"

OpenCL_Interface::OpenCL_Interface() 
{
    std::cout << "Empty constructor called" << std::endl;
}

// ====================================================

OpenCL_Interface::OpenCL_Interface(
    const int& targetPlatformIndex, 
    const int& targetDeviceIndex
)
{
    // get vector of platforms
    cl::Platform::get(&platforms);

    // abort if no platforms found
    assert(("No platforms found", platforms.size() > 0));

    // get target platform
    platform = platforms[targetPlatformIndex];

    // get vector of devices
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // abort if no devices found
    assert(("No devices found", devices.size() > 0));

    // get target device
    device = devices[targetDeviceIndex];

    // Output success to user
    std::cout << "Connected to accelerator device: ";
    std::cout << device.getInfo<CL_DEVICE_VENDOR>() << ", " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;

    // create context
    context = cl::Context({ device });

    // create queue
    queue = cl::CommandQueue(context, device);
}

// ====================================================

void OpenCL_Interface::addKernel(
    const std::string& kernelPath, 
    const std::string& kernelName,
    const std::vector<int>& inputArraySizes,
    const std::vector<int>& outputArraySizes
)
{
    OpenCL_KernelInterface kernelInterface;

    kernelInterface.numInputArrays = inputArraySizes.size();
    kernelInterface.numOutputArrays = outputArraySizes.size();

    kernelInterface.inputArraySizes = inputArraySizes;
    kernelInterface.outputArraySizes = outputArraySizes;

    // get kernel source
    std::string kernelSource = getKernelSource(kernelPath);

    // get kernel
    kernelInterface.kernel = getKernel(kernelSource, kernelName);

    // create read buffer vector
    kernelInterface.inputBuffers = {}; // empty vector

    for (int i = 0; i < kernelInterface.numInputArrays; ++i)
    {
        int numArrayElements = kernelInterface.inputArraySizes[i];

        cl::Buffer buffer(context, CL_MEM_READ_ONLY, sizeof(float) * numArrayElements);

        kernelInterface.inputBuffers.push_back(buffer);
    }

    // create write buffer vector
    kernelInterface.outputBuffers = {}; // empty vector

    for (int i = 0; i < kernelInterface.numOutputArrays; ++i)
    {
        int numArrayElements = kernelInterface.outputArraySizes[i];

        cl::Buffer buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * numArrayElements);

        kernelInterface.outputBuffers.push_back(buffer);
    }

    kernelInterfaces.push_back(kernelInterface);
}

// ====================================================

void OpenCL_Interface::runKernel(
    const int& targetKernelIndex,
    const int& numElements,
    const int& workgroupSize,
    const std::vector<float*>& inputArrays, 
    const std::vector<float*>& outputArrays
)
{
    // add input to input buffer
    for (int i = 0; i < kernelInterfaces[targetKernelIndex].numInputArrays; ++i)
    {
        float* array = inputArrays[i];

        queue.enqueueWriteBuffer(kernelInterfaces[targetKernelIndex].inputBuffers[i], CL_TRUE, 0, sizeof(float) * kernelInterfaces[targetKernelIndex].inputArraySizes[i], array);
    }

    // set kernel arguments
    for (int i = 0; i < kernelInterfaces[targetKernelIndex].numInputArrays; ++i)
    {
        kernelInterfaces[targetKernelIndex].kernel.setArg(i, kernelInterfaces[targetKernelIndex].inputBuffers[i]);
    }
    for (int i = 0; i < kernelInterfaces[targetKernelIndex].numOutputArrays; ++i)
    {
        kernelInterfaces[targetKernelIndex].kernel.setArg(i + kernelInterfaces[targetKernelIndex].numInputArrays, kernelInterfaces[targetKernelIndex].outputBuffers[i]);
    }

    // run kernel
    queue.enqueueNDRangeKernel(kernelInterfaces[targetKernelIndex].kernel, cl::NullRange, cl::NDRange(numElements), cl::NDRange(workgroupSize));

    // wait for kernel to finish
    queue.finish();

    // transfer output buffer to output
    for (int i = 0; i < kernelInterfaces[targetKernelIndex].numOutputArrays; ++i)
    {
        float* array = outputArrays[i];

        queue.enqueueReadBuffer(kernelInterfaces[targetKernelIndex].outputBuffers[i], CL_TRUE, 0, sizeof(float) * kernelInterfaces[targetKernelIndex].outputArraySizes[i], array);
    }
}

// ====================================================

// create kernel object given kernel source file
cl::Kernel OpenCL_Interface::getKernel(
    const std::string& kernelSource, 
    const std::string& kernelName
)
{
    // initialise program sources, takes array with const char* c string and length
    cl::Program::Sources sources;
    sources.push_back({ kernelSource.c_str(), kernelSource.length() });

    // build program
    cl::Program program(context, sources);

    assert(("Kernel did not build", program.build({ device }) == CL_SUCCESS));

    // create kernel object
    cl::Kernel kernel = cl::Kernel(program, kernelName.c_str());
    
    return kernel;
}

// ====================================================

// create kernel source given path
std::string OpenCL_Interface::getKernelSource(
    const std::string& kernelPath
)
{
    // create kernel source string
    std::ifstream file(kernelPath);
    std::string input;
    std::string kernelSource;

    while (file >> input)
    {
        input.append(" ");
        kernelSource.append(input);
    }

    return kernelSource;
}
