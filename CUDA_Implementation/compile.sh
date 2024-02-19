export LD_LIBRARY_PATH="../lib/SDL_install/lib"
module load cuda/10.0
nvcc main.cu -o DWT -I../lib/SDL_install/include -L../lib/SDL_install/lib -lSDL2