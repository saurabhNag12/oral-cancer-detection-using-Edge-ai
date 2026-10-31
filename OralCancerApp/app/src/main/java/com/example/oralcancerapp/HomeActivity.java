package com.example.oralcancerapp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class HomeActivity extends AppCompatActivity {
    private SessionManager session;

    private ImageView previewImage;
    private ImageView highlightImage;
    private TextView highlightTitle;
    private TextView resultText;
    private TextView confidenceText;
    private TextView modelText;
    private TextView messageText;
    private LinearLayout resultCard;
    private ProgressBar progress;

    private Uri cameraUri;
    private File selectedFile;

    private ActivityResultLauncher<Uri> cameraLauncher;
    private ActivityResultLauncher<String> galleryLauncher;
    private ActivityResultLauncher<String> cameraPermissionLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        session = new SessionManager(this);
        if (!session.loggedIn()) {
            openLogin();
            return;
        }

        setContentView(R.layout.activity_home);

        previewImage = findViewById(R.id.previewImage);
        highlightImage = findViewById(R.id.highlightImage);
        highlightTitle = findViewById(R.id.highlightTitle);
        resultText = findViewById(R.id.resultText);
        confidenceText = findViewById(R.id.confidenceText);
        modelText = findViewById(R.id.modelText);
        messageText = findViewById(R.id.messageText);
        resultCard = findViewById(R.id.resultCard);
        progress = findViewById(R.id.progress);

        TextView welcome = findViewById(R.id.welcome);
        welcome.setText(
                session.name().isEmpty()
                        ? "CancerTrack"
                        : "Hello, " + session.name()
        );

        Button captureButton = findViewById(R.id.captureButton);
        Button browseButton = findViewById(R.id.browseButton);
        Button esp32Button = findViewById(R.id.esp32Button);
        Button predictButton = findViewById(R.id.predictButton);
        Button logoutButton = findViewById(R.id.logoutButton);

        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                success -> {
                    if (success && cameraUri != null && selectedFile != null) {
                        previewImage.setImageURI(cameraUri);
                        clearPrediction();
                    }
                }
        );

        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        try {
                            selectedFile = copyUriToCache(uri, "gallery_image.jpg");
                            previewImage.setImageURI(uri);
                            clearPrediction();
                        } catch (Exception e) {
                            Toast.makeText(
                                    this,
                                    "Could not read selected image: " + e.getMessage(),
                                    Toast.LENGTH_LONG
                            ).show();
                        }
                    }
                }
        );

        cameraPermissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                granted -> {
                    if (granted) {
                        launchCamera();
                    } else {
                        Toast.makeText(
                                this,
                                "Camera permission is required to capture a photo",
                                Toast.LENGTH_LONG
                        ).show();
                    }
                }
        );

        captureButton.setOnClickListener(v -> capturePhoto());
        browseButton.setOnClickListener(v -> galleryLauncher.launch("image/*"));
        esp32Button.setOnClickListener(v -> captureEsp32());
        predictButton.setOnClickListener(v -> uploadPrediction());
        logoutButton.setOnClickListener(v -> logout());
    }

    private void capturePhoto() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED) {
            launchCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void launchCamera() {
        try {
            File dir = new File(getCacheDir(), "images");
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IllegalStateException("Unable to create camera cache");
            }

            selectedFile = new File(
                    dir,
                    "camera_" + System.currentTimeMillis() + ".jpg"
            );

            cameraUri = FileProvider.getUriForFile(
                    this,
                    getPackageName() + ".fileprovider",
                    selectedFile
            );

            cameraLauncher.launch(cameraUri);
        } catch (Exception e) {
            Toast.makeText(
                    this,
                    "Camera error: " + e.getMessage(),
                    Toast.LENGTH_LONG
            ).show();
        }
    }

    private File copyUriToCache(Uri uri, String name) throws Exception {
        File dir = new File(getCacheDir(), "images");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Unable to create image cache");
        }

        File out = new File(
                dir,
                System.currentTimeMillis() + "_" + name
        );

        try (
                InputStream input = getContentResolver().openInputStream(uri);
                FileOutputStream output = new FileOutputStream(out)
        ) {
            if (input == null) {
                throw new IllegalArgumentException("Unable to open image");
            }
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
        }

        return out;
    }

    private void captureEsp32() {
        setBusy(true);

        new Thread(() -> {
            try {
                Request request = new Request.Builder()
                        .url(Config.ESP32_CAPTURE_URL)
                        .get()
                        .build();

                try (okhttp3.Response response = ApiClient.http()
                        .newCall(request).execute()) {

                    if (!response.isSuccessful() || response.body() == null) {
                        throw new IllegalStateException(
                                "ESP32-CAM returned HTTP " + response.code()
                        );
                    }

                    byte[] bytes = response.body().bytes();

                    File dir = new File(getCacheDir(), "images");
                    if (!dir.exists() && !dir.mkdirs()) {
                        throw new IllegalStateException("Unable to create cache");
                    }

                    selectedFile = new File(
                            dir,
                            "esp32_" + System.currentTimeMillis() + ".jpg"
                    );

                    try (FileOutputStream out = new FileOutputStream(selectedFile)) {
                        out.write(bytes);
                    }

                    Bitmap bitmap = BitmapFactory.decodeByteArray(
                            bytes, 0, bytes.length
                    );

                    runOnUiThread(() -> {
                        setBusy(false);
                        previewImage.setImageBitmap(bitmap);
                        clearPrediction();
                        Toast.makeText(
                                this,
                                "ESP32-CAM image captured",
                                Toast.LENGTH_SHORT
                        ).show();
                    });
                }

            } catch (Exception e) {
                runOnUiThread(() -> {
                    setBusy(false);
                    Toast.makeText(
                            this,
                            "ESP32-CAM error: " + e.getMessage(),
                            Toast.LENGTH_LONG
                    ).show();
                });
            }
        }).start();
    }

    private void uploadPrediction() {
        if (selectedFile == null || !selectedFile.isFile()) {
            Toast.makeText(
                    this,
                    "Capture or browse an oral image first",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        setBusy(true);
        clearPrediction();

        RequestBody imageBody = RequestBody.create(
                selectedFile,
                MediaType.parse("image/*")
        );

        MultipartBody.Part part = MultipartBody.Part.createFormData(
                "image",
                selectedFile.getName(),
                imageBody
        );

        ApiClient.service()
                .predict(session.bearer(), part)
                .enqueue(new Callback<Models.PredictionResponse>() {
                    @Override
                    public void onResponse(
                            Call<Models.PredictionResponse> call,
                            Response<Models.PredictionResponse> response
                    ) {
                        setBusy(false);

                        if (response.code() == 401) {
                            session.clear();
                            openLogin();
                            return;
                        }

                        if (!response.isSuccessful()
                                || response.body() == null
                                || response.body().prediction == null) {
                            Toast.makeText(
                                    HomeActivity.this,
                                    "Prediction failed. Check Flask server/model files.",
                                    Toast.LENGTH_LONG
                            ).show();
                            return;
                        }

                        showPrediction(response.body());
                    }

                    @Override
                    public void onFailure(
                            Call<Models.PredictionResponse> call,
                            Throwable t
                    ) {
                        setBusy(false);
                        Toast.makeText(
                                HomeActivity.this,
                                "Server connection failed: " + t.getMessage(),
                                Toast.LENGTH_LONG
                        ).show();
                    }
                });
    }

    private void showPrediction(Models.PredictionResponse wrapper) {
        Models.Prediction p = wrapper.prediction;

        resultCard.setVisibility(View.VISIBLE);
        resultText.setText(
                "Predicted Result: "
                        + (p.is_cancer ? "CANCER / SUSPICIOUS" : "NO CANCER")
        );
        confidenceText.setText(
                String.format("Confidence: %.2f%%", p.confidence * 100.0)
        );
        modelText.setText(
                "Model: " + (
                        wrapper.model == null
                                ? "Best model"
                                : wrapper.model
                )
        );
        messageText.setText(p.message == null ? "" : p.message);

        if (p.is_cancer && p.highlight_url != null && !p.highlight_url.isEmpty()) {
            highlightTitle.setVisibility(View.VISIBLE);
            highlightImage.setVisibility(View.VISIBLE);
            loadProtectedImage(p.highlight_url, highlightImage);
        } else {
            highlightTitle.setVisibility(View.GONE);
            highlightImage.setVisibility(View.GONE);
        }
    }

    private void loadProtectedImage(String relativeUrl, ImageView target) {
        String absolute = relativeUrl.startsWith("http")
                ? relativeUrl
                : Config.BASE_URL.substring(0, Config.BASE_URL.length() - 1)
                    + (relativeUrl.startsWith("/") ? relativeUrl : "/" + relativeUrl);

        new Thread(() -> {
            try {
                Request request = new Request.Builder()
                        .url(absolute)
                        .header("Authorization", session.bearer())
                        .get()
                        .build();

                try (okhttp3.Response response = ApiClient.http()
                        .newCall(request).execute()) {

                    if (!response.isSuccessful() || response.body() == null) {
                        throw new IllegalStateException(
                                "Highlight HTTP " + response.code()
                        );
                    }

                    byte[] bytes = response.body().bytes();
                    Bitmap bitmap = BitmapFactory.decodeByteArray(
                            bytes, 0, bytes.length
                    );

                    runOnUiThread(() -> target.setImageBitmap(bitmap));
                }
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(
                        this,
                        "Could not load attention highlight: " + e.getMessage(),
                        Toast.LENGTH_LONG
                ).show());
            }
        }).start();
    }

    private void clearPrediction() {
        resultCard.setVisibility(View.GONE);
        highlightTitle.setVisibility(View.GONE);
        highlightImage.setVisibility(View.GONE);
        highlightImage.setImageDrawable(null);
    }

    private void setBusy(boolean busy) {
        progress.setVisibility(busy ? View.VISIBLE : View.GONE);
    }

    private void logout() {
        ApiClient.service().logout(session.bearer())
                .enqueue(new Callback<Models.MessageResponse>() {
                    @Override
                    public void onResponse(
                            Call<Models.MessageResponse> call,
                            Response<Models.MessageResponse> response
                    ) {
                        session.clear();
                        openLogin();
                    }

                    @Override
                    public void onFailure(
                            Call<Models.MessageResponse> call,
                            Throwable t
                    ) {
                        session.clear();
                        openLogin();
                    }
                });
    }

    private void openLogin() {
        Intent intent = new Intent(this, LoginActivity.class);
        intent.addFlags(
                Intent.FLAG_ACTIVITY_NEW_TASK
                        | Intent.FLAG_ACTIVITY_CLEAR_TASK
        );
        startActivity(intent);
        finish();
    }
}
