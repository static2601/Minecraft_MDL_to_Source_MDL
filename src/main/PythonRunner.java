package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class PythonRunner {

    private String[] logs = {"",""};
    public PythonRunner() {}
    public String[] PythonRun(String model_name, Map<String, String> args) throws IOException, InterruptedException {

        // Define the command and arguments for the Python script
        String modelname = model_name;
        String modname = args.get("modname");
        String assets = "assets\\assets";
        String path =   System.getProperty("user.dir") + "\\";
        String tools = args.get("tools");
        String game = args.get("game");
        String output = "assets\\mcexports";
        String scale = args.get("scale");
        String skybox_scale = args.get("skybox_scale");
        String compile_skybox = args.get("compile_skybox"); // key is '--compile-skybox' if true
        //String compile_models = args.get("compile_models");

        ProcessBuilder processBuilder = new ProcessBuilder(
                "python3",
                "assets\\mcexport.py",
                "--tools", tools,
                "--game", game,
                "--assets", path + assets,
                "--mod", modname,
                "--out", path + output,
                "--scale", scale,
                "--skybox-scale", skybox_scale,
                "--compile-skybox", compile_skybox,
                modelname
        );
        processBuilder.redirectErrorStream(true); // Merges stderr into stdout
        Process process = processBuilder.start();
        // Read the output from the Python script
        List<String> results = readProcessOutput(process.getInputStream());
        // Wait for the process to complete and get the exit code
        int exitCode = process.waitFor();

        logger("[------------------" + modelname + "------------------]", 1);
        logger("Mod: "+ modname, 1);
        logger("Game: "+ game, 1);
        //if (compile_skybox.equals("true"))
        logger("Scale: "+ scale, 1);
        //else logger("Scale: "+ scale, 1);
        logger("Python script output", 1);
        results.forEach((n) -> logger(n, 1));
        logger("Exit code: " + exitCode, 1);

        return logs;
    }

    private static List<String> readProcessOutput(java.io.InputStream inputStream) throws IOException {
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream))) {
            return bufferedReader.lines().collect(Collectors.toList());
        }
    }

    public void logger(String str, int priority) {
        logs[1] += str + "\n";
    }
}