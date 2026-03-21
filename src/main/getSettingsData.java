package main;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class getSettingsData {
    public ArrayList<JSONObject> settings = new ArrayList<>();
    //public ArrayList<JSONObject> defaults = new ArrayList<>();
    //private ArrayList<String> jar_paths = new ArrayList<>();
    private JSONObject jar_paths_map = new JSONObject();
    public GUIStart guiStartRef;
    private JSONObject jSettings;
    private JSONObject jModPaths;
    //private JSONArray jArrSettings;
    private JSONObject jDefaults;
    private String steamappsPath;
    private String file_chooser_dir;
    private String jar_chooser_dir;
    private JSONObject jar_adder_dir;
    private String shortGame;
    private String modName;
    private long scale;
    private long skybox_scale;
    private String tf2_tools_path;
    private String tf2_game_path;
    // defaults
    private String gmod_tools_path;
    private String gmod_game_path;
    private String css_tools_path;
    private String css_game_path;
    private String csgo_tools_path;
    private String csgo_game_path;
    private String l4d_tools_path;
    private String l4d_game_path;
    private String l4d2_tools_path;
    private String l4d2_game_path;
    private final String UserDir = System.getProperty("user.dir");
    private final File settingsJson = new File(UserDir + "\\assets\\settings.json");

    public getSettingsData() throws IOException, ParseException {
        // initialize data
        getJsonData();

        // get groups of data
        settings.forEach(s -> {
            //System.out.println(s.get("settings"));
            this.jSettings = (JSONObject) s.get("settings");
            this.jDefaults = (JSONObject) s.get("defaults");
            this.jModPaths = (JSONObject) s.get("mod_paths");
        });

        //System.out.println(jSettings.size());
        // settings object
        // define our variables
        try{
            // settings
            this.jar_chooser_dir = getObjKey("jar_chooser_dir").toString();

            //this.jar_paths = (ArrayList<String>) getObjKey("jar_paths");
            this.jar_paths_map.putAll((HashMap<String, String>) getObjHashMap());
            this.steamappsPath = getObjKey("steamappsPath").toString();
            this.file_chooser_dir = getObjKey("file_chooser_dir").toString();
            this.shortGame = getObjKey("shortGame").toString();
            this.modName = getObjKey("modName").toString();
            this.scale = (long) getObjKey("scale");
            this.skybox_scale = (long) getObjKey("skybox_scale");
            // defaults
            this.tf2_tools_path = getObjKey2("tf2:tools_path").toString();
            this.tf2_game_path = getObjKey2("tf2:game_path").toString();
            this.gmod_tools_path = getObjKey2("gmod:tools_path").toString();
            this.gmod_game_path = getObjKey2("gmod:game_path").toString();
            this.css_tools_path = getObjKey2("css:tools_path").toString();
            this.css_game_path = getObjKey2("css:game_path").toString();
            this.csgo_tools_path = getObjKey2("csgo:tools_path").toString();
            this.csgo_game_path = getObjKey2("csgo:game_path").toString();
            this.l4d_tools_path = getObjKey2("l4d:tools_path").toString();
            this.l4d_game_path = getObjKey2("l4d:game_path").toString();
            this.l4d2_tools_path = getObjKey2("l4d2:tools_path").toString();
            this.l4d2_game_path = getObjKey2("l4d2:game_path").toString();

        } catch (Exception e) {
            //System.out.println("getSettingsData(): " + Arrays.toString(e.getStackTrace()));
            e.printStackTrace();
        }

        //this.writeJson();
    }
    public void set_defaults() {

        this.jar_chooser_dir = "";
        //this.jar_paths = new ArrayList<>();
        this.jar_paths_map = new JSONObject();
        this.steamappsPath = "";
        this.file_chooser_dir = "";
        this.shortGame = "";
        this.modName = "";
        this.scale = 48;
        this.skybox_scale = 16;

        this.tf2_tools_path = "\\common\\Team Fortress 2\\bin";
        this.tf2_game_path = "\\common\\Team Fortress 2\\tf";

        this.gmod_tools_path = "\\common\\GarrysMod\\bin";
        this.gmod_game_path = "\\common\\GarrysMod\\garrysmod";

        this.css_tools_path = "\\common\\Counter-Strike Source\\bin";
        this.css_game_path = "\\common\\Counter-Strike Source\\cstrike";

        this.csgo_tools_path = "\\common\\Counter-Strike Global Offensive\\csgo\\bin";
        this.csgo_game_path = "\\common\\Counter-Strike Global Offensive\\csgo";

        this.l4d_tools_path = "\\common\\left 4 dead\\bin";
        this.l4d_game_path = "\\common\\left 4 dead\\left4dead";

        this.l4d2_tools_path = "\\common\\left 4 dead 2\\bin";
        this.l4d2_game_path = "\\common\\left 4 dead 2\\left4dead2";
    }
    public void writeJson() {
        //Write JSON file
        String UserDir = System.getProperty("user.dir");
        Path path = new File(UserDir + "\\Assets\\settings.json").toPath();
        try {
            // set all settings to settings.json
            JSONObject allSettings = new JSONObject();

            JSONObject settings = new JSONObject();
            settings.put("steamappsPath", this.steamappsPath);
            settings.put("jar_chooser_dir", this.jar_chooser_dir);
            settings.put("jar_adder_dir", this.jar_adder_dir);
            settings.put("file_chooser_dir", this.file_chooser_dir);
            settings.put("shortGame", this.shortGame);
            settings.put("modName", this.modName);
            settings.put("scale", this.scale);
            settings.put("skybox_scale", this.skybox_scale);
            //settings.put("jar_paths", this.jar_paths);
            //settings.put("jar_paths_map", this.jar_paths_map);
            allSettings.put("settings", settings);

            JSONObject defaults = new JSONObject();
            defaults.put("tf2:tools_path", this.tf2_tools_path);
            defaults.put("tf2:game_path", this.tf2_game_path);
            defaults.put("gmod:tools_path", this.gmod_tools_path);
            defaults.put("gmod:game_path", this.gmod_game_path);
            defaults.put("css:tools_path", this.css_tools_path);
            defaults.put("css:game_path", this.css_game_path);
            defaults.put("csgo:tools_path", this.csgo_tools_path);
            defaults.put("csgo:game_path", this.csgo_game_path);
            defaults.put("l4d:tools_path", this.l4d_tools_path);
            defaults.put("l4d:game_path", this.l4d_game_path);
            defaults.put("l4d2:tools_path", this.l4d2_tools_path);
            defaults.put("l4d2:game_path", this.l4d2_game_path);
            allSettings.put("defaults", defaults);

            JSONObject modPaths = new JSONObject();
            modPaths.putAll(this.jar_paths_map);
            //modPaths.putAll(this.jar_paths_map);
            allSettings.put("mod_paths", modPaths);


            final Gson gson = new GsonBuilder()
                    .setPrettyPrinting()
                    .create();
            String toSafe = gson.toJson(allSettings);
            Files.write(path, toSafe.getBytes());

            //file.write(allSettings.toJSONString());
            //file.flush();

        } catch (IOException e) {
            System.out.println("writeJson() in  getSettingsData");
            e.printStackTrace();
        }
    }

    //  get saved data from settings group
    public boolean has_key(String key) {
        return jSettings.containsKey(key);
    }
    public ArrayList<File> get_jar_paths() {
        JSONArray list = (JSONArray) this.jSettings.get("jar_paths");
        ArrayList<File> files = new ArrayList<>();
        for (Object f : list) {
            files.add(new File(f.toString()));
        }
        return files;
        //return (ArrayList<File>) this.jSettings.get("jar_paths");
    }
//    /// add val to jar_paths list in settings.json, if it doesn't exist already
//    public void set_jar_paths(String val) {
//        if (!this.jar_paths.contains(val)) {
//            this.jar_paths.add(val);
//            this.writeJson();
//        }
//    }
    public String get_name_in_jar_paths(String name) {
        if (this.jar_paths_map.containsKey(name)) {
            return this.jar_paths_map.get(name).toString();
        }
        return "";
    }
    public ArrayList<File> get_jar_paths_map() {
        ArrayList<File> arr = new ArrayList<>();
        this.jar_paths_map.values().forEach(val ->{
            File file = new File(val.toString());
            arr.add(file);
        });
        return arr;
    }
    public void set_jar_paths_map(String name, String jar_path) {
        if (!this.jar_paths_map.containsKey(name)) {
            this.jar_paths_map.put(name, jar_path);
            this.writeJson();
        }
    }
    public void remove_jar_paths_map(String name) {
        if (this.jar_paths_map.containsKey(name)) {
            this.jar_paths_map.remove(name);
            this.writeJson();
        }
    }
    public String get_jar_chooser_dir() {

        return this.jSettings.get("jar_chooser_dir").toString();
    }
    public void set_jar_chooser_dir(String val) {
        this.jar_chooser_dir = val;
        this.writeJson();
    }
    public String get_file_chooser_dir() {

        return this.jSettings.get("file_chooser_dir").toString();
    }
    public void set_file_chooser_dir(String val) {
        this.file_chooser_dir = val;
        this.writeJson();
    }
    public String get_steamappsPath() {

        return this.jSettings.get("steamappsPath").toString();
    }
    public void set_steamappsPath(String val) {
        this.steamappsPath = val;
        this.writeJson();
    }
    public String get_shortGame() {
        return this.jSettings.get("shortGame").toString();
    }
    public void set_shortGame(String val) {
        this.shortGame = val;
        this.writeJson();
    }
    public long get_scale() {
        return (long) this.jSettings.get("scale");
    }
    public void set_scale(long val) {
        this.scale = val;
        this.writeJson();
    }
    public long get_skybox_scale() {
        return (long) this.jSettings.get("skybox_scale");
    }
    public void set_skybox_scale(long val) {
        this.skybox_scale = val;
        this.writeJson();
    }
    public String get_game_path(String game) {
        return jDefaults.get(game + ":game_path").toString();
    }
    public String get_tools_path(String game) {
        return jDefaults.get(game + ":tools_path").toString();
    }

    /// get json save data from file system
    public void getJsonData() throws IOException, ParseException {
        // parsing file "JSONExample.json"
        //File file = new File("assets/settings.json");
//        String UserDir = System.getProperty("user.dir");
//        File settingsJson = new File(UserDir + "\\assets\\settings.json");

        // check if settings.json exists first
        if (settingsJson.exists()) {
            JSONObject obj = (JSONObject) new JSONParser().parse(
                    new FileReader(settingsJson));
            if (obj != null)
                settings.add(obj);
        }
        else {
            // create file and write defaults
            System.out.println("Creating new settings.json");
            set_defaults();
            writeJson();
            JSONObject obj = (JSONObject) new JSONParser().parse(
                    new FileReader(settingsJson));
            if (obj != null)
                settings.add(obj);
        }
        //settings.forEach(System.out::println);
    }

    public Object getObjKey(String name) {
        return this.jSettings.get(name);
    }
    public Object getObjHashMap() {
        return this.jModPaths;
    }
    public Object getObjKey2(String name) {
        return this.jDefaults.get(name);
    }
}
