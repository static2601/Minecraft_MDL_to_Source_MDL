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

    private String[] logs = {"","","","",""};
    public PythonRunner() {}
    public String[] PythonRun(String model_name, Map<String, String> args) throws IOException, InterruptedException {

        String UserDir = System.getProperty("user.dir");
        String modname = "";

        // argument for --assets
        // full path to the 'assets' folder within project's assets_dir
        // example use in python: textures_path: str = os.path.join(assets_path(mod_assets_dir), mod_name, "textures", "block")
        // was path_to
        String base_assets_dir = UserDir + File.separator + "assets";

        // needs to be model name (no .json) with base being models/modelname.
        // any sub folders after 'model/' need included with name
        String modelname = "";

        // model_name is from blockstates folder
        // looks like 'namespace:block/off/model'
        if (model_name.contains(":")) {
            String[] split = model_name.split(":");
            modname = split[0];
            modelname = split[1].replace("block/", "");
        }

        // need what variables still?
        // assets folder should be shared?
        // which assets folder should be used??? sourcecraft or this one?? or somewhere else?
        // using our assets by default. should check if Sourcecraft has one set.

        String fs = File.separator;
        String path = UserDir + fs;
        String tools = args.get("tools");
        String game = args.get("game");
        String mcJar = args.get("mc_jar");
        String modJar = args.get("mod_jar");
        String output = "assets"+ fs +"mcexports";
        String scale = args.get("scale");
        String skybox_scale = args.get("skybox_scale");
        String compile_skybox = args.get("compile_skybox"); // key is '--compile-skybox' if true

        ProcessBuilder processBuilder = new ProcessBuilder(
                "python3",
                "assets"+ fs +"mcexport.py",
                "--tools", tools,
                "--game", game,
                "--mcjar", mcJar,
                "--mod_jar", modJar,
                "--assets", base_assets_dir,
                "--mod", modname,
                "--out", path + output,
                "--scale", scale,
                "--skybox-scale", skybox_scale,
                "--compile-skybox", compile_skybox,
                "--allow-template",
                modelname
        );

        System.out.println("[PythonRun] start processing python");
        processBuilder.redirectErrorStream(true); // Merges stderr into stdout
        Process process = processBuilder.start();
        // Read the output from the Python script
        System.out.println(process.getInputStream());
        List<String> results = readProcessOutput(process.getInputStream());

        // Wait for the process to complete and get the exit code
        int exitCode = process.waitFor();
        System.out.println("[PythonRun] exitCode: " + exitCode);

        logger("[------------------" + modelname + "------------------]", 1);
        logger("Mod: "+ modname, 1);
        logger("Game: "+ game, 1);
        logger("Scale: "+ scale, 1);
        logger("Python script output", 1);

        // print to logger python script results
        ArrayList<String> status_results = new ArrayList<>();
        ArrayList<String> console_results = new ArrayList<>();
        ArrayList<String> build_results = new ArrayList<>();
        ArrayList<String> json_paths = new ArrayList<>();

        System.out.println("[PythonRun] results.size(): "+ results.size());

        boolean in_build_logs = false;
        for (String line : results) {
            if (line.contains("Start Build Log")) {
                in_build_logs = true;
                continue;
            }
            if (line.contains("End Build Log")) {
                in_build_logs = false;
                continue;
            }
            if (in_build_logs) {
                build_results.add(line);
                continue;
            }
            if (line.contains("[INFO]")) {
                status_results.add(line);
            }
            if (line.contains("[JSON_Path]:")) {
                json_paths.add(
                        line.substring(line.indexOf("[JSON_Path]:"))
                        .replace("[JSON_Path]:", ""));
            }
            if (!(line.contains("Start Build Log") || line.contains("End Build Log"))) {
                if (line.contains("[DEBUG]") || line.contains("[WARNING]") || line.contains("[INFO]") || line.contains("[ERROR]")) {
                    // everything minus build log
                    console_results.add(line);
                }
            }
        }
        System.out.println("[PythonRun] for each loop done");

        status_results.forEach((n) -> logger(n, 0)); // only status info
        console_results.forEach((n) -> logger(n, 1)); // everything minus build results
        build_results.forEach((n) -> logger(n, 2)); // only build results
        results.forEach((n) -> logger(n, 3)); // everything
        json_paths.forEach((n) -> logger(n, 4)); // json paths

        // exit code for python script
        logger("Exit code: " + exitCode, 1);
        logger("Exit code: " + exitCode, 3);

        System.out.println("[PythonRun] return logs");
        return logs;
    }

    private static List<String> readProcessOutput(java.io.InputStream inputStream) throws IOException {
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream))) {
            return bufferedReader.lines().collect(Collectors.toList());
        }
    }

    /// logger(str: str, priority: int (0 = status, 1 = console, 2 = build, 3 = everything))
    public void logger(String str, int priority) {
        logs[priority] += str + "\n";
    }
}