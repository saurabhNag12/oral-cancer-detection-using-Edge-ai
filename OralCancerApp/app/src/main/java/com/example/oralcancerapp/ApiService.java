package com.example.oralcancerapp;

import java.util.Map;
import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Header;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface ApiService {
    @POST("api/auth/signup")
    Call<Models.AuthResponse> signup(@Body Map<String, String> body);

    @POST("api/auth/login")
    Call<Models.AuthResponse> login(@Body Map<String, String> body);

    @POST("api/auth/logout")
    Call<Models.MessageResponse> logout(
            @Header("Authorization") String authorization
    );

    @Multipart
    @POST("api/predictions")
    Call<Models.PredictionResponse> predict(
            @Header("Authorization") String authorization,
            @Part MultipartBody.Part image
    );
}
