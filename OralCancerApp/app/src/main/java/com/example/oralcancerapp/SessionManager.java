package com.example.oralcancerapp;

import android.content.Context;
import android.content.SharedPreferences;

public class SessionManager {
    private static final String PREF = "cancertrack_session";
    private static final String TOKEN = "token";
    private static final String NAME = "name";

    private final SharedPreferences preferences;

    public SessionManager(Context context) {
        preferences = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
    }

    public void save(String token, String name) {
        preferences.edit()
                .putString(TOKEN, token)
                .putString(NAME, name)
                .apply();
    }

    public String token() {
        return preferences.getString(TOKEN, "");
    }

    public String name() {
        return preferences.getString(NAME, "");
    }

    public boolean loggedIn() {
        return !token().isEmpty();
    }

    public void clear() {
        preferences.edit().clear().apply();
    }

    public String bearer() {
        return "Bearer " + token();
    }
}
