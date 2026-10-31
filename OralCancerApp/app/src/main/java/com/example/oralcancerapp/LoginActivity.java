package com.example.oralcancerapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.HashMap;
import java.util.Map;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class LoginActivity extends AppCompatActivity {
    private EditText email;
    private EditText password;
    private ProgressBar progress;
    private SessionManager session;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        session = new SessionManager(this);
        if (session.loggedIn()) {
            openHome();
            return;
        }

        setContentView(R.layout.activity_login);

        email = findViewById(R.id.email);
        password = findViewById(R.id.password);
        progress = findViewById(R.id.progress);

        Button loginButton = findViewById(R.id.loginButton);
        Button signupButton = findViewById(R.id.signupButton);

        loginButton.setOnClickListener(v -> login());
        signupButton.setOnClickListener(v ->
                startActivity(new Intent(this, SignupActivity.class))
        );
    }

    private void login() {
        String e = email.getText().toString().trim();
        String p = password.getText().toString();

        if (e.isEmpty() || p.isEmpty()) {
            Toast.makeText(this, "Enter email and password", Toast.LENGTH_SHORT).show();
            return;
        }

        progress.setVisibility(View.VISIBLE);

        Map<String, String> body = new HashMap<>();
        body.put("email", e);
        body.put("password", p);

        ApiClient.service().login(body).enqueue(new Callback<Models.AuthResponse>() {
            @Override
            public void onResponse(
                    Call<Models.AuthResponse> call,
                    Response<Models.AuthResponse> response
            ) {
                progress.setVisibility(View.GONE);

                if (response.isSuccessful() && response.body() != null) {
                    Models.AuthResponse data = response.body();
                    String name = data.user != null ? data.user.name : "";
                    session.save(data.token, name);
                    openHome();
                } else {
                    Toast.makeText(
                            LoginActivity.this,
                            "Login failed. Check email/password and server.",
                            Toast.LENGTH_LONG
                    ).show();
                }
            }

            @Override
            public void onFailure(Call<Models.AuthResponse> call, Throwable t) {
                progress.setVisibility(View.GONE);
                Toast.makeText(
                        LoginActivity.this,
                        "Cannot connect to server: " + t.getMessage(),
                        Toast.LENGTH_LONG
                ).show();
            }
        });
    }

    private void openHome() {
        Intent intent = new Intent(this, HomeActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }
}
