#pragma once
#include<stdio.h>
#include <SDL2/SDL.h>

unsigned char* loadImage(const char *filename,int*width,int *height){

    SDL_Surface* surface = SDL_LoadBMP(filename);

    if (!surface) {
        fprintf(stderr, "Error loading image: %s\n", SDL_GetError());
        return NULL;
    }

    *width = surface->w;
    *height = surface->h;

    unsigned char* image = (unsigned char*)malloc(*width * *height * sizeof(unsigned char));

    if (!image) {
        fprintf(stderr, "Error allocating memory for image\n");
        SDL_FreeSurface(surface);
        return NULL;
    }

    // Copy pixel values from SDL surface to the image array
    for (int i = 0; i < *height; i++) {
        for (int j = 0; j < *width; j++) {
            Uint8 pixelValue = *((Uint8*)surface->pixels + i * surface->pitch + j);
            image[i * *width + j] = pixelValue;
        }
    }

    SDL_FreeSurface(surface);

    return image;
}


SDL_Texture* createTextureFromfloatArray(SDL_Renderer* renderer, float* imageData, int width, int height) {
    // Create a surface to hold the image data
    SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, 32,
        0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);

    if (!surface) {
        printf("Unable to create surface! SDL Error: %s\n", SDL_GetError());
        return NULL;
    }

    // Copy the float array data to the surface
    Uint32* pixels = (Uint32*)surface->pixels;
    for (int i = 0; i < width * height; ++i) {
        Uint8 value = (Uint8)(imageData[i]);

        // Since it's a single channel, we use the value for R, G, B, and A
        pixels[i] = SDL_MapRGBA(surface->format, value, value, value, 255);
    }

    // Create a texture from the surface
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        printf("Unable to create texture from surface! SDL Error: %s\n", SDL_GetError());
    }

    // Free the surface as it's no longer needed
    SDL_FreeSurface(surface);

    return texture;
}


void displayImage(float* imageData, int width, int height) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL Error: %s\n", SDL_GetError());
        return;
    }

    // Create a window and renderer
    SDL_Window* window = SDL_CreateWindow("SDL Image from float Array", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 800, 600, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Window could not be created! SDL Error: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        printf("Renderer could not be created! SDL Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    // Create a texture from the float array
    SDL_Texture* texture = createTextureFromfloatArray(renderer, imageData, width, height);
    if (!texture) {
        // Handle the error as needed
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

     SDL_Event event;
    int quit = 0;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = 1;
            }
        }

        // Render the texture
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    // Cleanup
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}


void saveImage(const char* filename, float* imageData, int width, int height) {
    SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);

    if (surface == NULL) {
        fprintf(stderr, "SDL_CreateRGBSurface failed: %s\n", SDL_GetError());
        return;
    }

    Uint32* pixels = (Uint32*)surface->pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Map the float value directly to Uint8 in the [0, 255] range
            Uint8 value = (Uint8) fmaxf(0.0f, fminf(imageData[y * width + x], 255.0f));
            pixels[y * width + x] = SDL_MapRGB(surface->format, value, value, value);
        }
    }

    if (SDL_SaveBMP(surface, filename) != 0) {
        fprintf(stderr, "SDL_SaveBMP failed: %s\n", SDL_GetError());
    }

    SDL_FreeSurface(surface);
}