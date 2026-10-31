package com.example.oralcancerapp;

public final class Config {
    private Config() {}

    // IMPORTANT:
    // Replace with your laptop/server IPv4 address on the same Wi-Fi/LAN.
    // The trailing slash is required by Retrofit.
    public static final String BASE_URL = "http://10.82.61.232:5000/";

    // Optional ESP32-CAM still-image endpoint.
    // Change this to your ESP32-CAM IP. CameraWebServer commonly exposes /capture.
    public static final String ESP32_CAPTURE_URL = "http://10.82.61.76/capture";
}
