#!/bin/bash
export LD_LIBRARY_PATH="../lib/SDL_install/lib"
g++ main.c -o DWT_sequential -I../lib/SDL_install/include -L../lib/SDL_install/lib -lSDL2
