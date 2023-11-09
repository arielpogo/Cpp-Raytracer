//Code written by Dr. Yicheng Tu

cudaEvent_t TIMING_START_EVENT, TIMING_STOP_EVENT;
float TIMING_ELAPSED_TIME;

void TIMING_START(){
    cudaEventCreate (&TIMING_START_EVENT);
    cudaEventCreate (&TIMING_STOP_EVENT);
    cudaEventRecord (TIMING_START_EVENT, 0);
}


void TIMING_STOP(){
    cudaEventRecord (TIMING_STOP_EVENT, 0);
    cudaEventSynchronize (TIMING_STOP_EVENT);
    cudaEventElapsedTime (&TIMING_ELAPSED_TIME, TIMING_START_EVENT, TIMING_STOP_EVENT);
    cudaEventDestroy (TIMING_START_EVENT);
    cudaEventDestroy (TIMING_STOP_EVENT);
}

void TIMING_PRINT(){
    std::cout << "Running time: " << TIMING_ELAPSED_TIME << "ms (" << TIMING_ELAPSED_TIME/1000.0f << " s)";
}