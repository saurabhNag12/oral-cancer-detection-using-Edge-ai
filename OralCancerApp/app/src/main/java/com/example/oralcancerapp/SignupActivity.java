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

public class SignupActivity extends AppCompatActivity {
    private EditText name;
    private EditText email;
    private EditText password;
    private ProgressBar progress;
    private SessionManager session;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_signup);

        session = new SessionManager(this);

        name = findViewById(R.id.name);
        email = findViewById(R.id.email);
        password = findViewById(R.id.password);
        progress = findViewById(R.id.progress);

        Button createButton = findViewById(R.id.createButton);
        Button backButton = findViewById(R.id.backButton);

        createButton.setOnClickListener(v -> signup());
        backButton.setOnClickListener(v -> finish());
    }

    private void signup() {
        String n = name.getText().toString().trim();
        String e = email.getText().toString().trim();
        String p = password.getText().toString();

        if (n.length() < 2 || e.isEmpty() || p.length() < 8) {
            Toast.makeText(
                    this,
                    "Enter name, valid email and password of at least 8 characters",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        progress.setVisibility(View.VISIBLE);

        Map<String, String> body = new HashMap<>();
        body.put("name", n);
        body.put("email", e);
        body.put("password", p);

        ApiClient.service().signup(body).enqueue(new Callback<Models.AuthResponse>() {
            @Override
            public void onResponse(
                    Call<Models.AuthResponse> call,
                    Response<Models.AuthResponse> response
            ) {
                progress.setVisibility(View.GONE);

                if (response.isSuccessful() && response.body() != null) {
                    Models.AuthResponse data = response.body();
                    String patientName = data.user != null ? data.user.name : n;
                    session.save(data.token, patientName);

                    Intent intent = new Intent(
                            SignupActivity.this,
                            HomeActivity.class
                    );
                    intent.addFlags(
                            Intent.FLAG_ACTIVITY_NEW_TASK
                                    | Intent.FLAG_ACTIVITY_CLEAR_TASK
                    );
                    startActivity(intent);
                    finish();
                } else {
                    Toast.makeText(
                            SignupActivity.this,
                            "Sign-up failed. The email may already be registered.",
                            Toast.LENGTH_LONG
                    ).show();
                }
            }

            @Override
            public void onFailure(Call<Models.AuthResponse> call, Throwable t) {
                progress.setVisibility(View.GONE);
                Toast.makeText(
                        SignupActivity.this,
                        "Cannot connect to server: " + t.getMessage(),
                        Toast.LENGTH_LONG
                ).show();
            }
        });
    }
}
