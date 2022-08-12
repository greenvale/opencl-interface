kernel void myKernel (
    global const float* A, 
    global const float* B,
    global float* C
)
{
    int index = get_global_id(0);

    C[index] = A[index] + B[index];
    
}