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
import java.util.Objects;

public class getSettingsData {
    public ArrayList<JSONObject> settings = new ArrayList<>();
    public GUIStart guiStartRef;
    private JSONObject jSettings;
    private String steamappsPath;
    private String file_chooser_dir;
    private String shortGame;
    private String modName;
    private Integer scale;
    private Integer skybox_scale;

    public getSettingsData() throws IOException, ParseException {
        // initialize data
        getJsonData();

        // get groups of data
        settings.forEach(s -> {
            System.out.println(s.get("settings"));
            this.jSettings = (JSONObject) s.get("settings");
        });
        //System.out.println(jSettings.size());
        // settings object
        // define our variables
        try{
            this.steamappsPath = getObjKey("steamappsPath").toString();
            this.file_chooser_dir = getObjKey("file_chooser_dir").toString();
            this.shortGame = getObjKey("shortGame").toString();
            this.modName = getObjKey("modName").toString();
            this.scale = (Integer) getObjKey("scale");
            this.skybox_scale = (Integer) getObjKey("skybox_scale");
        } catch (Exception e) {
            System.out.println(e);
        }

        //this.writeJson();
    }

    public void writeJson() {
        //Write JSON file
        String UserDir = System.getProperty("user.dir");
        Path path = new File(UserDir + "\\Assets\\settings.json").toPath();
        try {
            JSONObject settings = new JSONObject();
            settings.put("steamappsPath", this.steamappsPath);
            settings.put("file_chooser_dir", this.file_chooser_dir);
            settings.put("shortGame", this.shortGame);
            settings.put("modName", this.modName);
            settings.put("scale", this.scale);
            settings.put("skybox_scale", this.skybox_scale);

            // get all texture settings
            JSONObject allSettings = new JSONObject();
            allSettings.put("settings", settings);

            final Gson gson = new GsonBuilder()
                    .setPrettyPrinting()
                    .create();

            String toSafe = gson.toJson(allSettings);

            Files.write(path, toSafe.getBytes());

            //file.write(allSettings.toJSONString());
            //file.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //  get saved data from settings group
    public boolean has_key(String key) {
        return jSettings.containsKey(key);
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
    public String get_modName() {
        return this.jSettings.get("modName").toString();
    }
    public void set_modName(String val) {
        this.modName = val;
        this.writeJson();
    }
    public Integer get_scale() {
        return (Integer) this.jSettings.get("scale");
    }
    public void set_scale(Integer val) {
        this.scale = val;
        this.writeJson();
    }
    public Integer get_skybox_scale() {
        return (Integer) this.jSettings.get("skybox_scale");
    }
    public void set_skybox_scale(Integer val) {
        this.skybox_scale = val;
        this.writeJson();
    }

    /// get json save data from file system
    public void getJsonData() throws IOException, ParseException {
        // parsing file "JSONExample.json"
        //File file = new File("assets/settings.json");
        String UserDir = System.getProperty("user.dir");
        JSONObject obj = (JSONObject) new JSONParser().parse(
                new FileReader(UserDir + "\\assets\\settings.json"));
        if (obj != null)
            settings.add(obj);
    }

    public Object getObjKey(String name) {
        return this.jSettings.get(name);
    }
}
