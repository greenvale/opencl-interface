kernel void myKernel (
    global const float* A, 
    global float* B
)
{
    int index = get_global_id(0);

    B[index] = 2.0 * A[index];
    
}