#include <CL/opencl.hpp>

#include "OpenCLInterface.hpp"

OpenCLInterface::OpenCLInterface() 
{
    std::cout << "Default constructor called" << std::endl;
}

// ====================================================

OpenCLInterface::OpenCLInterface(
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

void OpenCLInterface::addKernel(
    const std::string& kernelPath, 
    const std::string& kernelName,
    const std::vector<int>& inputArraySizes,
    const std::vector<int>& outputArraySizes
)
{
    // -------------------------------------------
    // KERNEL CREATION

    // get kernel source
    std::string kernelSource = getKernelSource(kernelPath);

    // get kernel
    kernel = getKernel(kernelSource, kernelName);

    // -------------------------------------------
    // BUFFER CREATION

    int numInputArrays = inputArraySizes.size();
    int numOutputArrays = outputArraySizes.size();

    // create read buffer vector
    readBuffers = {}; // empty vector

    for (int i = 0; i < numInputArrays; ++i)
    {
        int numArrayElements = inputArraySizes[i];

        cl::Buffer buffer(context, CL_MEM_READ_ONLY, sizeof(float) * numArrayElements);

        readBuffers.push_back(buffer);
    }

    // create write buffer vector
    writeBuffers = {}; // empty vector

    for (int i = 0; i < numOutputArrays; ++i)
    {
        int numArrayElements = outputArraySizes[i];

        cl::Buffer buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * numArrayElements);

        writeBuffers.push_back(buffer);
    }
}

// ====================================================

void OpenCLInterface::runKernel(
    const int& numElements,
    const int& workgroupSize,
    const std::vector<std::tuple<float*, int>>& inputArrays, 
    const std::vector<std::tuple<float*, int>>& outputArrays
)
{
    int numInputArrays = inputArrays.size();
    int numOutputArrays = outputArrays.size();

    // -------------------------------------------
    // KERNEL RUNTIME

    // add input to read buffer
    for (int i = 0; i < numInputArrays; ++i)
    {
        float* array = std::get<0>(inputArrays[i]);
        int numArrayElements = std::get<1>(inputArrays[i]);

        queue.enqueueWriteBuffer(readBuffers[i], CL_TRUE, 0, sizeof(float) * numArrayElements, array);
    }

    for (int i = 0; i < numInputArrays; ++i)
    {
        kernel.setArg(i, readBuffers[i]);
    }
    for (int i = 0; i < numOutputArrays; ++i)
    {
        kernel.setArg(i + numInputArrays, writeBuffers[i]);
    }

    // run kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numElements), cl::NDRange(workgroupSize));

    // wait for kernel to finish
    queue.finish();

    // transfer write buffer to output
    for (int i = 0; i < numOutputArrays; ++i)
    {
        float* array = std::get<0>(outputArrays[i]);
        int numArrayElements = std::get<1>(outputArrays[i]);

        queue.enqueueReadBuffer(writeBuffers[i], CL_TRUE, 0, sizeof(float) * numArrayElements, array);
    }
}

// ====================================================

// create kernel object given kernel source file
cl::Kernel OpenCLInterface::getKernel(
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
std::string OpenCLInterface::getKernelSource(
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
