package com.example.kaki;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Controller
public class KakiController {

    // 常駐しているPython(FastAPI)サーバーのURL
    // Renderに両方デプロイする場合は、PythonサーバーのURLをここに設定する
    // (例: 環境変数から読み込むようにしてもよい)
    private static final String PYTHON_API_URL =
            System.getenv().getOrDefault("KAKI_API_URL", "http://localhost:8000") + "/analyze";

    private static final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping("/upload")
    public String handleFileUpload(@RequestParam("file") MultipartFile file, Model model) {
        if (file.isEmpty()) {
            model.addAttribute("message", "画像を選択してください");
            return "index";
        }

        try {
            String fileName = file.getOriginalFilename();
            if (fileName == null || fileName.isEmpty()) fileName = "upload.jpg";

            // multipart/form-data で常駐サーバーに画像をそのまま転送する
            String boundary = "----KakiBoundary" + System.currentTimeMillis();
            byte[] multipartBody = buildMultipartBody(boundary, fileName, file.getBytes());

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(PYTHON_API_URL))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .timeout(Duration.ofSeconds(20))
                    .POST(HttpRequest.BodyPublishers.ofByteArray(multipartBody))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            String body = response.body();

            // 簡易JSONパース(依存を増やしたくない場合。Jacksonが使えるなら差し替え推奨)
            String resultImage = extractJsonValue(body, "resultImage");
            String message = extractJsonValue(body, "message");

            if (resultImage != null) {
                model.addAttribute("resultImage", resultImage);
            }
            model.addAttribute("message", message != null ? message : "処理が完了しました");

        } catch (Exception e) {
            e.printStackTrace();
            model.addAttribute("message", "システムエラー: " + e.getMessage());
        }

        return "index";
    }

    private byte[] buildMultipartBody(String boundary, String fileName, byte[] fileBytes) throws Exception {
        String header = "--" + boundary + "\r\n" +
                "Content-Disposition: form-data; name=\"file\"; filename=\"" + fileName + "\"\r\n" +
                "Content-Type: application/octet-stream\r\n\r\n";
        String footer = "\r\n--" + boundary + "--\r\n";

        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        out.write(header.getBytes());
        out.write(fileBytes);
        out.write(footer.getBytes());
        return out.toByteArray();
    }

    private String extractJsonValue(String json, String key) {
        // "key":"value" 形式の簡易抽出(base64データURLのカンマ等も壊さないよう非貪欲マッチ)
        Pattern pattern = Pattern.compile("\"" + key + "\"\\s*:\\s*\"(.*?)\"(?=\\s*[,}])");
        Matcher matcher = pattern.matcher(json);
        return matcher.find() ? matcher.group(1) : null;
    }
}
