package com.example.oralcancerapp;

import com.google.gson.annotations.SerializedName;
import java.util.Map;

public final class Models {
    private Models() {}

    public static class User {
        public int id;
        public String name;
        public String email;
    }

    public static class AuthResponse {
        public String token;
        public double expires_at;
        public User user;
    }

    public static class Prediction {
        public int id;
        public String created_at;

        @SerializedName("class")
        public String className;

        public double confidence;
        public boolean is_cancer;
        public Map<String, Double> probabilities;
        public String original_url;
        public String highlight_url;
        public String message;
        public String reported_stage;
        public String stage_note;
    }

    public static class PredictionResponse {
        public Prediction prediction;
        public String model;
    }

    public static class MessageResponse {
        public String message;
        public String error;
    }
}
