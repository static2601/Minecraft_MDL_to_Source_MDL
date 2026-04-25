package main;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class JarExtractor {
    private static int no_blockstates_count = 0;

    public static String extractFileFromJar(String jarFilePath, String fileNameToExtract, String destinationDir) {
        try (JarFile jarFile = new JarFile(jarFilePath)) {
            fileNameToExtract = fileNameToExtract.replace("\\", "/");
            // Find the specific entry
            JarEntry entry = jarFile.getJarEntry(fileNameToExtract);

            if (entry == null) {
                System.out.println("File not found in the JAR: " + fileNameToExtract + ", JAR: "+ jarFilePath);
                jarFile.close();
                return null;
            }

            // Define the output file path
            File outputFile = new File(destinationDir, entry.getName());

            // Ensure the destination directory structure is created
            File parent = outputFile.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }

            // Read the data from the JAR entry and write it to the new file
            try (InputStream is = jarFile.getInputStream(entry);
                 FileOutputStream fos = new FileOutputStream(outputFile)) {

                byte[] buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, bytesRead);

                }
                System.out.println("Successfully extracted: " + fileNameToExtract + " to " + outputFile.getAbsolutePath());
                jarFile.close();
                return outputFile.getAbsolutePath();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
    public static void getModIdsFromJars() {

    }
    public static int getNoBlockstatesCount() {
        return no_blockstates_count;
    }
    /// go through jar to find the mod's modId. Not the same as getting from folder name.
    private static String getModIdFromJar(File jar_path) {
        //TODO check the dependencies of the mod and see if the mod is within the modslist and not
        // process models that dont have all dependencies available. would have to run after modlist
        // as been processed so we can check against the list of modIds.
        try (JarFile jarFile = new JarFile(jar_path)) {
            Enumeration<JarEntry> entries = jarFile.entries();

            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                if (entry.getName().endsWith("neoforge.mods.toml")) {
                    // open ...mods.json and look for...
                    // JSONObject key: id, get value
                    // may not contain id, go off of name. check both
                    //  "manifestType": "minecraftModpack",
                    //  "manifestVersion": 1,
                    //  "name": "New Simple Mods [FABRIC] - MC 1.21.1 ",
                    //if (checkForBlockStates(jar_path)) {
                    String prefix = "";
                    if (!checkForBlockStates(jar_path)) prefix = "nb|";
                    String mod_line = readFileInJar(jar_path, entry.getName(), "modId");
                    if (!mod_line.isEmpty()) {
                        //System.out.println("[getModIdFromJar()] mod_line(neoforge.mods.toml) as \"modId:\" " + mod_line);
                        String mod_id = mod_line.substring(mod_line.indexOf("\"") + 1, mod_line.lastIndexOf("\""));
                        jarFile.close();
                        return prefix + mod_id;
                    }
                }
                else
                if (entry.getName().endsWith("mods.toml")) {
                    // open ...mods.json and look for...
                    // JSONObject key: id, get value
                    // may not contain id, go off of name. check both
                    //  "manifestType": "minecraftModpack",
                    //  "manifestVersion": 1,
                    //  "name": "New Simple Mods [FABRIC] - MC 1.21.1 ",
                    //if (checkForBlockStates(jar_path)) {
                    String prefix = "";
                    if (!checkForBlockStates(jar_path)) prefix = "nb|";
                    String mod_line = readFileInJar(jar_path, entry.getName(), "modId");
                    if (!mod_line.isEmpty()) {
                    //System.out.println("[getModIdFromJar()] mod_line(mods.toml) as \"modId:\" " + mod_line);
                    String mod_id = mod_line.substring(mod_line.indexOf("\"") + 1, mod_line.lastIndexOf("\""));
                    jarFile.close();
                    return prefix + mod_id;
                    }
                }
                else
                if (entry.getName().endsWith("fabric.mod.json")) {
                    // open ...mods.json and look for...
                    // JSONObject key: id, get value
                    // {
                    //  "schemaVersion": 1,
                    //  "id": "oakores",
                    //  "version": "1.1.0",
                    //
                    //  "name": "Oak's Ore Mod",
                    //if (checkForBlockStates(jar_path)) {
                    String prefix = "";
                    if (!checkForBlockStates(jar_path)) prefix = "nb|";
                    String mod_line = readFileInJar(jar_path, entry.getName(), "id:");

                    if (!mod_line.isEmpty()) {
                        String mod_id = mod_line
                                .split(":")[1]
                                .substring(mod_line.indexOf("\"") + 1, mod_line.lastIndexOf("\""));
                        System.out.println("[getModIdFromJar()] mod_line(fabric.mod.json) as \"id:\" " + mod_line);
                        //System.out.println("[getModIdFromJar()] mod_line(fabric.mod.json) as 'name:' " + mod_line);
                        jarFile.close();
                        return prefix + mod_id;
                    }
                    //}
                }
                else if (entry.getName().endsWith("manifest.json")) {
                    // open ...mods.json and look for...
                    // JSONObject key: id, get value
                    // may not contain id, go off of name. check both
                    //  "manifestType": "minecraftModpack",
                    //  "manifestVersion": 1,
                    //  "name": "New Simple Mods [FABRIC] - MC 1.21.1 ",

                    //TODO
                    // make add other non blockstate mods to a list, make all mods added with modId to refer
                    // back to for importing textures/models
                    //if (checkForBlockStates(jar_path)) {
                    String prefix = "";
                    if (!checkForBlockStates(jar_path)) prefix = "nb|";
                    String mod_line = readFileInJar(jar_path, entry.getName(), "modId");
                    System.out.println("[getModIdFromJar()] mod_line as \"modId:\" " + mod_line);
                    if (mod_line.isEmpty()) {
                        mod_line = readFileInJar(jar_path, entry.getName(), "id:");
                        System.out.println("[getModIdFromJar()] mod_line as \"id:\" " + mod_line);
                        String mod_id = mod_line
                                .split(":")[1]
                                .substring(mod_line.indexOf("\"") + 1, mod_line.lastIndexOf("\""));
                        jarFile.close();
                        return prefix + mod_id;
                    }
                    else {
                        String mod_id = mod_line
                                .split(":")[1]
                                .substring(mod_line.indexOf("\"") + 1, mod_line.lastIndexOf("\""));
                        System.out.println("[getModIdFromJar()] mod_line(manifest.json) as \"registry_name:\" " + mod_line);
                        jarFile.close();
                        return prefix + mod_id;
                    }
                    //}
                }
                else if (entry.getName().contains("com/mojang/")) {
                    // open ...mods.json and look for...
                    // JSONObject key: id, get value
                    // may not contain id, go off of name. check both
                    //  "manifestType": "minecraftModpack",
                    //  "manifestVersion": 1,
                    //  "name": "New Simple Mods [FABRIC] - MC 1.21.1 ",
                    //if (checkForBlockStates(jar_path)) {
                    String prefix = "";
                    if (!checkForBlockStates(jar_path)) prefix = "nb|";
//                    System.out.println("[getModNameFromJar] entry.getName().contains(\"com/mojang/\"). jar name: "
//                            + jar_path.getName() + ", entry name: " + entry.getName());
//                        String mod_line = readFileInJar(jar_path, entry.getName(), "modId");
//                        System.out.println("mod_line: " + mod_line);
//                        String mod_id = mod_line.substring(mod_line.indexOf("\"")+1, mod_line.lastIndexOf("\""));
//                        System.out.println("mod_id: "+ mod_id);
                    jarFile.close();
                    return prefix + "minecraft";
                    //}
                }
                else {
                    // all non blockstates add with prefix to seperate from others
                    //TODO - remake cleaner getModIdFromJar()
                    //System.out.println("[getModNameFromJar] nothing found in jar_path: " + jar_path);

                }
            }
            // nothing found for conditions set
            //if (checkForBlockStates(jar_path)) {
            String prefix = "";
            if (!checkForBlockStates(jar_path)) prefix = "nb|";
            if (jar_path.getName().contains("-") && jar_path.getName().contains(".jar")) {
                System.out.println("[getModNameFromJar] Cant find mod id, setting jar name. jar name:" + jar_path.getName());
                jarFile.close();
                return prefix + jar_path.getName().split("-")[0];
            }
            else {
                System.out.println("[getModNameFromJar] No '-' found to split by. jar name: " + jar_path.getName());
                jarFile.close();
                return prefix + jar_path.getName();
            }
            //}
            //no_blockstates_count++;
            //System.out.println("[getModNameFromJar] No blockstates directory. jar name: " + jar_path.getName());
            //return "";
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static String readFileInJar(File jarFilePath, String fileNameToRead, String searchText) {
        try (JarFile jarFile = new JarFile(jarFilePath)) {
            // Find the specific entry
            JarEntry entry = jarFile.getJarEntry(fileNameToRead);

            if (entry == null) {
                System.out.println("[readFileInJar] File not found in the JAR: " + fileNameToRead);
                jarFile.close();
                return "";
            }
            // Read the data from the JAR entry and write it to the new file
            try (InputStream is = jarFile.getInputStream(entry);
                 //InputStream is = JarExtractor.class.getResourceAsStream(filename);
                 InputStreamReader isr = new InputStreamReader(is, StandardCharsets.UTF_8);
                 BufferedReader br = new BufferedReader(isr)) {

                if (is == null) {
                    System.out.println("[readFileInJar] File not found: " + fileNameToRead);
                    jarFile.close();
                    return "";
                }

                //StringBuilder sb = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.contains(searchText)) {
                        jarFile.close();
                        return line;
                    }
                    //sb.append(line).append("\n"); // Append newline character if needed
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
//        System.out.println("[readFileInJar] nothing found. jar_path: " + jarFilePath
//                + ", fileNameToRead: " + fileNameToRead
//                + ", searchText: " + searchText);
        return "";
    }

    /// return if blockstates is a directory in jar
    public static boolean checkForBlockStates(File jar_path) {
        //System.out.println("checking for blockstates in jar: "+ jar_path);
        try (JarFile jarFile = new JarFile(jar_path)) {
            Enumeration<JarEntry> entries = jarFile.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                // Check if the entry is a directory
                if (entry.getName().contains("blockstates")) {
                    //System.out.println("blockstates path: "+entry.getName());
                    jarFile.close();
                    return true;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    /// return all jsons of blockstates folder in jar. json: 'assets/mod/blockstates/blockstate_name.json'
    public static ArrayList<String> getBlockStatesOfMod(String mod, File jar_path) {
        // since modListMaps key is mod/subfolder now, will have to do this different
        if (mod.contains("/"))
            mod = mod.split("/")[1];
        ArrayList<String> blockStatesJSONS = new ArrayList<>();
        try (JarFile jarFile = new JarFile(jar_path)) {
            Enumeration<JarEntry> entries = jarFile.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                // Check if the entry is a directory
                if (entry.getName().startsWith("assets/"+ mod + "/blockstates/")) {
                    //System.out.println(entry.getName());
                    String blockstatesDir = "assets/"+ mod + "/blockstates/";
                    //System.out.println("blockstatesDir.toString(): "+blockstatesDir);
                    //String json_name = "";
                    if (entry.getName().split(blockstatesDir).length > 1) {
                        //String json_name = entry.getName().split(blockstatesDir)[1];
                        //blockStatesJSONS.add(entry.getName());
                        //blockStatesJSONS.add(json_name);
                        blockStatesJSONS.add(entry.toString());
                    }
                }
            }
            //return blockStatesJSONS;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return blockStatesJSONS;
    }
    public static ArrayList<String> getModelsOfMod(String mod, File jar_path) {
        // since modListMaps key is mod/subfolder now, will have to do this different
        if (mod.contains("/"))
            mod = mod.split("/")[1];
        ArrayList<String> modelsJSONS = new ArrayList<>();
        try (JarFile jarFile = new JarFile(jar_path)) {
            Enumeration<JarEntry> entries = jarFile.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                // Check if the entry is a directory
                if (entry.getName().startsWith("assets/"+ mod + "/models/")) {
                    //System.out.println(entry.getName());
                    String modelsDir = "assets/"+ mod + "/models/";
                    //System.out.println("blockstatesDir.toString(): "+blockstatesDir);
                    //String json_name = "";
                    if (entry.getName().split(modelsDir).length > 1) {
                        String json_name = entry.getName().split(modelsDir)[1];
                        //blockStatesJSONS.add(entry.getName());
                        //blockStatesJSONS.add(json_name);
                        modelsJSONS.add(entry.toString());
                    }
                }
            }
            //return modelsJSONS;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return modelsJSONS;
    }
    /// get hashmap of value to place in gui lists box. will set to settings and return mod_map=key:modId, value:jar path
    public static HashMap<String, File> getModList(File jars_path) {
        HashMap<String, File> mods_map = new HashMap<>();
        //ArrayList<String> mod_list = new ArrayList<>();
        getSettingsData settings = GUIStart.settings;
        if (jars_path.isFile()) {
            if (jars_path.isFile() && jars_path.getName().endsWith(".jar")) {
                // for every jar coming in, need to get modId from meta-inf
                // need to check dependencies, set an array to each jar to check later or can get later?
                // any
                //settings.set_jar_paths(jars_path.toString());
                String modId = getModIdFromJar(jars_path); // <-- change this will only return string
                //ArrayList<String> modId = extractModNameFromJar(jars_path);

                ArrayList<String> dirs = extractModNameFromJar(jars_path);
                String suffix;
                if (!modId.isEmpty()) {
                    // set jar to settings.json
                    //settings.set_jar_paths(jars_path.toString());
                    HashMap<String, String> jar_map = new HashMap<>();
                    //TODO
                    jar_map.put(modId, jars_path.toString());
                    settings.set_jar_paths_map(modId, jars_path.toString());
                    for (String name : dirs) {
                        // if name in dirs does is not the same
                        // put as eg: mekanism/minecraft else put just as mekanism
                        if (!modId.equals(name)) {
                            suffix = "/" + name;
                        }
                        else suffix = "";
                        mods_map.put(modId + suffix, jars_path.getAbsoluteFile());
                    }
                }
            }
        }
//        else if (jars_path.isDirectory()) {
//            for (File f : Objects.requireNonNull(jars_path.listFiles())) {
//                if (f.isFile() && f.getName().endsWith(".jar")) {
//                    //settings.set_jar_paths(f.toString());
//                    String results = getModNameFromJar(f); // <-- change this
//                    ArrayList<String> dirs = extractModNameFromJar(f);
//                    String suffix;
//                    if (!results.isEmpty()) {
//                        // set jar to settings.json
//                        settings.set_jar_paths(jars_path.toString());
//                        for (String name : dirs) {
//                            // if name in dirs does is not the same
//                            // put as eg: mekanism-minecraft else put just as mekanism
//                            if (!results.equals(name)) {
//                                suffix = "/" + name;
//                            }
//                            else suffix = "";
//                            mods_map.put(results + suffix, f.getAbsoluteFile());
//                        }
//                        //if (dirs.size() > 1) results + "-" +
//                        //mods_map.put(results + suffix, f.getAbsoluteFile());
//                        //System.out.println(results + "\n\tjar: " + f.getName());
//                    }
//                }
//            }
//        }
        return mods_map;
    }
    /// get folders in jar's 'assets' containing a 'blockstates' directory. returns ArrayList of Strings
    public static ArrayList<String> extractModNameFromJar(File jar) {
        ArrayList<String> mod_names = new ArrayList<>();
        try (JarFile jarFile = new JarFile(jar)) {
            Enumeration<JarEntry> entries = jarFile.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                // Check if the entry is a directory
                if (entry.getName().startsWith("assets/") && entry.getName().contains("blockstates/")) {
                    String modname = entry.getName().split("/")[1];
                    //System.out.println(modname);
                    if (!check_in_array(mod_names, modname)) {
                        mod_names.add(modname);
                    }
                }
            }
            jarFile.close();
            return mod_names;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return mod_names;
    }
    private static boolean check_in_array(ArrayList<String> arr, String name) {
        for (String s : arr) {
            if (s.equals(name)) return true;
        }
        return false;
    }
}
