package main;

import com.formdev.flatlaf.FlatDarkLaf;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;
import org.json.simple.parser.ParseException;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.util.*;
import java.util.List;

import static main.JarExtractor.*;

public class GUIStart {
    public static getSettingsData settings;
    private JPanel jPanel1;
    private JPanel main_jPanel;
    private JButton model_selection_btn;
    private JTextField model_txtfield;
    private JButton convert_btn;
    private JButton console_clear_btn;
    private JSpinner model_scale_spinner;
    private JRadioButton make_skybox_models_checkbox;
    private JSpinner skybox_scale_spinner;
    private JScrollPane output_panel;
    public JTextArea console_output;
    private JTextField steamapps_txtfield;
    private JButton steamapps_dir_search_btn;
    private JComboBox game_combobox;
    private JComboBox mod_combobox;
    private JButton open_json_files_btn;
    private JRadioButton make_models_checkbox;
    private JButton open_materials_btn;
    private JTabbedPane tabbedPane1;
    private JTextArea status_output;
    private JScrollPane status_panel;
    private JScrollPane build_panel;
    private JTextArea build_output;
    private JComboBox model_log_selector_cb;
    private JLabel model_log_label;
    private JPanel console_jpanel;
    private JTabbedPane tabbedPane2;
    private JList jars_list;
    private JList jar_files_list;
    private JButton jar_add_button;
    private JButton jar_sort_button;
    private JButton jar_remove_button;
    private JLabel availableModsLabel;
    private JCheckBox variantsCheckBox;
    private JCheckBox modelsCheckBox;
    private JButton btn_sourcecraft_import;
    private JButton btn_geometry_import;
    private File file_chooser_dir;
    private File jar_chooser_dir;
    private File jar_adder_dir;
    private final String UserDir;
    private String console_log;
    private String status_log;
    private String build_log;
    private String assets_dir;

    private ArrayList<HashMap<String, String>> model_files;
    private ArrayList<String> build_logs;
    private ArrayList<String> console_logs;
    private ArrayList<ArrayList<String>> modelJsons;
    private ArrayList<String> currentModelJsons;
    public BlockStates BLOCKSTATES;
    private DefaultListModel<String> listModel = new DefaultListModel<>();
    private DefaultListModel<String> listModel2 = new DefaultListModel<>();
    private HashMap<String, File> modListMaps = new HashMap<>();
    /// hashmap key: blockstate name as listed in gui list, value: full path to blockstate_name.json from assets/ of jar
    private HashMap<String, String> modBSFileListMaps = new HashMap<>();
    /// full file path to 'sourcecraft_importing.json'. empty acts as false when converting.
    private File sourcecraft_importing = new File("");
    private File[] geometry_importing = new File[]{};

    static void main(String[] args) throws IOException, ParseException {
        FlatDarkLaf.setup();
        new GUIStart();
    }
    public GUIStart() throws IOException, ParseException {

        JFrame frame = new JFrame();
        frame.setContentPane(jPanel1);
        frame.setTitle("MC to Source Engine Mdl Convertor");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setSize(900, 650);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        settings = new getSettingsData();
        model_files = new ArrayList<>();
        build_logs = new ArrayList<>();
        console_logs = new ArrayList<>();
        modelJsons = new ArrayList<>();
        currentModelJsons = new ArrayList<>();

        settings.guiStartRef = this;

        status_log = "";
        console_log = "";
        build_log = "";
        UserDir = System.getProperty("user.dir");
        model_scale_spinner.setValue(48);
        skybox_scale_spinner.setValue(16);

        // reload saved data
        //System.out.println("settings.settings.tostring: "+ settings.settings);
        //settings.settings.forEach(t -> System.out.println(t.keySet()));

        try {
            steamapps_txtfield.setText(settings.get_steamappsPath());
            game_combobox.setSelectedItem(settings.get_shortGame());
            model_scale_spinner.setValue(settings.get_scale());
            skybox_scale_spinner.setValue(settings.get_skybox_scale());
            setToModList(settings.get_jar_paths_map());
        }
        catch (Exception e) {
            //System.out.println("GUIStart(): " + Arrays.toString(e.getStackTrace()));
            e.printStackTrace();
        }

        // listeners
        convert_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (steamapps_txtfield.getText().isEmpty()) {
                    logger("Steamapps Path must be filled.", 0);
                    return;
                }
                if (geometry_importing.length > 0) {
                    model_files.clear();

                    // [runPython] model_map: variant_model, minecraft:block/azalea
                    // [runPython] model_map: mod_jar, C:\Users\statiic\AppData\Roaming\.minecraft\versions\1.21.11\1.21.11.jar
                    // [runPython] model_map: model_mod_dir, minecraft
                    // [runPython] model_map: variant_model_path, E:\Coding Projects\ProjectFiles\IdeaProjects\Minecraft MDL to Source MDL\assets\assets\minecraft\models\block\azalea.json
                    // [runPython] model_map: blockstates_model, E:\Coding Projects\ProjectFiles\IdeaProjects\Minecraft MDL to Source MDL\assets\assets\minecraft\blockstates\azalea.json

                    for (File geo_model : geometry_importing) {
//                        HashMap<String, String> model = new HashMap<>();
//                        model.put("variant_model", k);
//                        model.put("mod_jar", "");
//                        model.put("model_mod_dir", "");
//                        model.put("variant_model_path", v.toString());
//                        model.put("blockstates_model", "");
//                        model_files.add(model)
                        HashMap<String, String> model_map = new HashMap<>();
                        //model_map.put("variant_model", block);
                        //model_map.put("variant_model_path", extracted_path2);
                        //model_map.put("model_mod_dir", block.split(":")[0]);
                        //model_map.put("mod_jar", jar_path);
                        //model_map.put("blockstates_model", "");

                        String mod_name = "";
                        if (geo_model.toString().contains("entity\\")) {
                            String fname = geo_model.toString().split("\\\\assets\\\\bedrock_assets\\\\")[1];
                            mod_name = fname.split("\\\\entity\\\\")[0];
                        }
//                        if (geo_model.toString().contains("models\\")) {
//                            String[] parts = geo_model.toString().split("models\\\\")[0].split("\\\\");
//                            mod_name = parts[parts.length - 1];
//                        }
                        System.out.println("geo_model: " + geo_model);
                        System.out.println("mod_name: " + mod_name);
                        String variant_model_parent = geo_model.getParent();
                        String variant_model = mod_name + ":" + geo_model.getName();
                        model_map.put("variant_model", variant_model.replace(".json", "")); // selected file minus the path to it,
                        model_map.put("model_mode", "geometry");
                        model_map.put("is_geo", "true");
                        model_map.put("model_mod_dir", mod_name);
                        model_map.put("variant_model_path", String.valueOf(geo_model)); // full path to json_model, of what is selected
                        // will only select one json, that json containing the path to the entity.geo.json
                        assets_dir = variant_model_parent;
                        // what is needed for python
                        // json_model: eg: minecraft:block/anvil
                        // model_path: os.path.join(mod_assets_path, mod_name, "models", "block", model_subfolders)
                        // mod_assets_path:
                        // mod_name:
                        // model_subfolders
                        // out_dir:
                        // others like skybox and units size
                        // os.path.join(out_dir, "modelsrc", mod_name, sb_dir, model_subfolders)


                        System.out.println("adding model: "+ model_map.get("variant_model") + " to model_files");
                        // adding model: minecraft:block/acacia_pressure_plate to model_files
                        // adding model: minecraft:block/acacia_pressure_plate_down to model_files
                        model_files.add(model_map);
                    }
                }
                else
                if (!sourcecraft_importing.toString().isEmpty()) {
                    model_files.clear();
                    Map<String, String> map = getJsonContent(sourcecraft_importing);
                    map.forEach((block, full_model_path) -> {
                        try {
                            HashMap<String, HashMap<String, String>> bs_maps = doBlockStates(new File(full_model_path), block);
                            for (String key : bs_maps.keySet()) {
                                HashMap<String, String> model = bs_maps.get(key);
                                System.out.println("adding model: "+ model.get("variant_model") + " to model_files");
                                // adding model: minecraft:block/acacia_pressure_plate to model_files
                                // adding model: minecraft:block/acacia_pressure_plate_down to model_files
                                model_files.add(model);
                            }
                        } catch (IOException ex) {
                            throw new RuntimeException(ex);
                        }
                    });
                }
                // convert paths for tools and game folders
                else
                if (!jar_files_list.getSelectedValuesList().isEmpty()) {
                    List<String> selected_items = jar_files_list.getSelectedValuesList();
                    model_files.clear();

                    for (String item : selected_items) {
                        File item_value = new File(modBSFileListMaps.get(item));
                        //item_value = new File(item_value.toString().split(":")[1]); //TODO

                        try {
                            HashMap<String, HashMap<String, String>> bs_maps = doBlockStates(item_value, "");
                            for (String key : bs_maps.keySet()) {
                                HashMap<String, String> model = bs_maps.get(key);
                                System.out.println("adding model: "+ model.get("variant_model") + " to model_files");
                                // adding model: minecraft:block/acacia_pressure_plate to model_files
                                // adding model: minecraft:block/acacia_pressure_plate_down to model_files
                                model_files.add(model);
                            }
                        } catch (IOException ex) {
                            throw new RuntimeException(ex);
                        }
                    }
                }
                else {
                    logger(" A JSON model must be selected.", 0);
                    return;
                }

                String steamapps_path = new File(steamapps_txtfield.getText()).getPath();
                String game_selection = game_combobox.getSelectedItem().toString().toLowerCase();

                String game_path = steamapps_path + settings.get_game_path(game_selection.toLowerCase());
                String tools_path = steamapps_path + settings.get_tools_path(game_selection.toLowerCase());
                logger("game path: " + game_path, 0);
                logger("tools path: " + tools_path, 0);
                logger("assets_dir path: " + assets_dir, 0);

                // save data to json
                settings.set_scale( (long) model_scale_spinner.getValue());
                settings.set_skybox_scale( (long) skybox_scale_spinner.getValue());
                settings.set_shortGame(game_combobox.getSelectedItem().toString());
                settings.writeJson();

                // run pythonrunner
                Map<String, String> args = new HashMap<>();
                args.put("assets", assets_dir);

                args.put("tools", tools_path);
                args.put("game", game_path);
                args.put("scale", model_scale_spinner.getValue().toString());
                args.put("skybox_scale", skybox_scale_spinner.getValue().toString());

                // if true, tell mcexports.py to use skybox scale instead of regular
                if (make_skybox_models_checkbox.isSelected())
                    args.put("compile_skybox", "true");
                else args.put("compile_skybox", "false");

                logger("--Conversion Started--",0);

                SwingWorker worker = new SwingWorker() {
                    @Override
                    protected Object doInBackground() throws Exception {

                        for (HashMap<String, String> mdl_map : model_files) {
                            System.out.println("mdl_map: "+ mdl_map);
                            runPython(mdl_map, args);
                        }
                        logger("--Conversions Done!--", 0);
                        return null;
                    }
                };
                worker.execute();
            }
        });
        steamapps_dir_search_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //logger(e.toString(), 0);
                File[] selectedFile = fileChooserDialog(false);
                if (selectedFile != null) {
                    String s = selectedFile[0].toString();
                    //logger(s, 0);
                    s = s.substring(0, s.indexOf("steamapps")+9);

                    //logger(s, 0);
                    steamapps_txtfield.setText(s);
                    settings.set_steamappsPath(s);
                }
            }
        });
        console_clear_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                status_output.setText("");
                console_output.setText("");
                build_output.setText("");
                console_log = "";
                status_log = "";
                build_log = "";
                build_logs.clear();
                modelJsons.clear();
                model_log_selector_cb.removeAllItems();
            }
        });
        open_json_files_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {

                int selectedIndex = model_log_selector_cb.getSelectedIndex();
                if (selectedIndex != -1) {
                    System.out.println("selectedIndex: " + selectedIndex);
                    ArrayList<String> filePaths = modelJsons.get(selectedIndex);
                    System.out.println(modelJsons);

                    //assert filePaths != null;
                    for (String model_path : filePaths) {
                        try {
                            System.out.println("model_path: " + model_path);
                            //ProcessBuilder pb = new ProcessBuilder("notepad.exe", model_path);
                            String sublime = "C:\\Program Files\\Sublime Text\\sublime_text.exe";
                            ProcessBuilder pb = new ProcessBuilder(sublime, model_path);
                            pb.start();
                        } catch (IOException e) {
                            System.err.println("An error occurred trying to open Notepad: " + e.getMessage());
                            e.printStackTrace();
                        }
                    }
                }
            }
        });
        make_models_checkbox.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                // checked
                if(e.getStateChange() == ItemEvent.SELECTED) {
                    make_models_checkbox.setSelected(true);
                    make_skybox_models_checkbox.setSelected(false);
                }
                else {
                    make_models_checkbox.setSelected(false);
                    make_skybox_models_checkbox.setSelected(true);
                }
            }
        });
        make_skybox_models_checkbox.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                // checked
                if(e.getStateChange() == ItemEvent.SELECTED) {
                    make_models_checkbox.setSelected(false);
                    make_skybox_models_checkbox.setSelected(true);
                }
                else {
                    make_models_checkbox.setSelected(true);
                    make_skybox_models_checkbox.setSelected(false);
                }
            }
        });
        open_materials_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                String steamapps_path = new File(steamapps_txtfield.getText()).getPath();
                String game_selection = game_combobox.getSelectedItem().toString().toLowerCase();

                int selectedIndex = model_log_selector_cb.getSelectedIndex();
                if (selectedIndex != -1) {
                    System.out.println("selectedIndex: " + selectedIndex);
                    //ArrayList<String> filePaths = modelJsons.get(selectedIndex);
                    System.out.println(modelJsons);

                    String tools_path = steamapps_path + settings.get_tools_path(game_selection.toLowerCase());
                    String mdl_path = "";

                    for (String line : build_logs.get(selectedIndex).split("\n")) {
                        if (line.contains("writing") && line.contains(".mdl:")) {
                            //TODO probably should have a better way of getting mdl_path
                            mdl_path = line
                                    .replace("writing ", "")
                                    .replace(".mdl:", ".mdl");

                        }
                    }
                    System.out.println(mdl_path);
                    if (!mdl_path.isEmpty()) {
                        File mdl_file = new File(mdl_path);
                        if (mdl_file.exists()) {
                            System.out.println("mdl_file: " + mdl_file);
                            System.out.println("hlmv.exe path: " + tools_path + "\\hlmv.exe");

                            try {
                                ProcessBuilder pb = new ProcessBuilder(tools_path + "\\hlmv.exe", mdl_file.toString());
                                pb.start();

                            } catch (IOException e) {
                                System.err.println("An error occurred trying to open HL:MV. Error: " + e.getMessage());
                                e.printStackTrace();
                            }
                        }
                        else System.out.println("MDL doesnt exist at path: " + mdl_file);
                    }
                }
            }
        });
        model_log_selector_cb.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int selectedIndex = model_log_selector_cb.getSelectedIndex();
                build_output.setText(build_logs.get(selectedIndex));
                console_output.setText(console_logs.get(selectedIndex));
                build_output.setCaretPosition(0);
                console_output.setCaretPosition(0);
            }
        });

        ActionListener listener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                if (e.getSource().equals(jar_add_button)) {
                    // add mod to jar_list
                    setToModList(JarAdderDialog());
                }
                if (e.getSource().equals(jar_remove_button)) {
                    // remove selected jar
                    String selected_value = jars_list.getSelectedValue().toString();
                    listModel.removeElement(selected_value);
                    settings.remove_jar_paths_map(selected_value);
                }
            }
        };
        jar_add_button.addActionListener(listener);
        jar_remove_button.addActionListener(listener);

        jars_list.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                if (jars_list.getSelectedValue() == null) return;
                //System.out.println("getValueIsAdjusting: " + jars_list.getSelectedValue());
                // with mod name selected get blockstates of mod

                String mod = jars_list.getSelectedValue().toString();
                File jar_path = modListMaps.get(mod);

                ArrayList<String> modBSFiles = getBlockStatesOfMod(mod, jar_path);
                //ArrayList<String> modModelFiles = getModelsOfMod(mod, jar_path);

                //System.out.println(modFiles.toString());
                //if (modFiles == null) return;
                listModel2.removeAllElements();
                modBSFileListMaps.clear();
                for (String f : modBSFiles) {
                    // get only the model.json
                    String json = f.substring(f.lastIndexOf("/")+1);
                    String json_name = json.replace(".json", "");
                    // needs to extract files to a temp folder to be converted
                    // or get a copy of the json and process, but would need to change how it converts model_files
                    modBSFileListMaps.put(json_name, f);
                    //modBSFileListMaps.put("bs:"+json_name, f);
                    //System.out.println("f: "+ f);
                    listModel2.addElement(json_name);
                }
//                for (String f : modModelFiles) {
//                    // get only the model.json
//                    String json = f.substring(f.lastIndexOf("/")+1);
//                    String json_name = json.replace(".json", "");
//                    // needs to extract files to a temp folder to be converted
//                    // or get a copy of the json and process, but would need to change how it converts model_files
//                    modBSFileListMaps.put("model:"+json_name, f);
//
//                    //System.out.println("f: "+ f);
//                    listModel2.addElement(json_name);
//                }
                jar_files_list.setModel(sortList(listModel2));
            }
        });
        jar_files_list.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                List<String> selected_items = jar_files_list.getSelectedValuesList();
//                selected_items.forEach((item) ->{
//                    System.out.println("key: " + item);
//                    System.out.println("value: " + modFileListMaps.get(item));
//                });
            }
        });
        ActionListener listener1 = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // not needed below, when updated, have list redo with/without lists
                if (e.getSource().equals(variantsCheckBox)) {

                }
                if (e.getSource().equals(modelsCheckBox)) {

                }
            }
        };
        variantsCheckBox.addActionListener(listener1);
        modelsCheckBox.addActionListener(listener1);

        btn_sourcecraft_import.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // open file chooser dialog to where sourcecraft import json should be
                File[] selectedFiles = fileChooserDialog(true);
                for (File file : selectedFiles) {
                    if (file != null) {
                        if (file.toString().contains("sourcecraft_import.json")) {
                            // run main
                            sourcecraft_importing = file;
                            convert_btn.doClick();
                            sourcecraft_importing = new File("");
                            return;


                        }
                    }
                }
            }
        });
        btn_geometry_import.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // open file chooser dialog to where sourcecraft import json should be
                File[] selectedFiles = fileChooserDialog(true);
                System.out.println("selectedFiles.toString():" + Arrays.toString(selectedFiles));

                geometry_importing = selectedFiles;
                System.out.println(Arrays.toString(geometry_importing));
                convert_btn.doClick();
                // clear after running so converted button if pressed doesn't redo this function
                // since it was last ran
                geometry_importing = new File[]{};
                return;

//                for (File file : selectedFiles) {
//                    if (file != null) {
//                        if (file.toString().contains(".json")) {
//                            // run main
//
//                            geometry_importing = file.listFiles();
//                            System.out.println(Arrays.toString(geometry_importing));
//                            convert_btn.doClick();
//                            //sourcecraft_importing = new File("");
//                            return;
//                        }
//                    }
//                }
            }
        });
    }


    /// set mods to list of mods in the gui
    private void setToModList(ArrayList<File> selectedFiles) {
        // add jar file
        //ArrayList<File> selectedFiles = JarAdderDialog();
        if (selectedFiles == null) return;
        ArrayList<File> directory_files = new ArrayList<>();
        for (File file : selectedFiles) {
            // file is .jar
            //settings.set_jar_paths(f.toString());
            if (file.isDirectory())
                directory_files.addAll(Arrays.asList(
                        Objects.requireNonNull(file.listFiles())));
            else directory_files.add(file);
        }
        for (File df : directory_files) {
            HashMap<String, File> map = getModList(df);
            for (Map.Entry<String, File> m : map.entrySet()) {
                // key could have multiple values
                //for (String s : m.getKey()) {
                if (!listModel.contains(m.getKey())) {
                    listModel.addElement(m.getKey());
                    // put mod_name : path_to_jar
                    modListMaps.put(m.getKey(), m.getValue());
                }
            }
        }
        // sort
        jars_list.setModel(sortList(listModel));
        System.out.println("["+getNoBlockstatesCount()+"] mods do not contain a blockstates directory. Ignoring...");
    }

    private DefaultListModel<String> sortList(DefaultListModel<String> list_model) {
        // sort
        ArrayList<String> temp_arr = new ArrayList<>();
        for (int i = 0; i < list_model.size(); i++) {
            temp_arr.add(list_model.get(i));
        }
        Collections.sort(temp_arr);
        list_model.removeAllElements();
        for (String s : temp_arr) {
            list_model.addElement(s);
            //System.out.println("s: "+ s + ", path to jar: " + modListMaps.get(s));
        }
        return list_model;
    }

    private void runPython(HashMap<String, String> model_map, Map<String, String> args) {
        // model_map.put("variant_model", mdl);
        // model_map.put("variant_model_path", extracted_path2);
        // model_map.put("model_mod_dir", mdl.split(":")[0]);
        // model_map.put("mod_jar", jar_path);
        // model_map.put("blockstates_model", model_path.toString());
        //System.out.println("[runPython] model_map: " + model_map);
        model_map.forEach((k,v) ->{
            System.out.println("[runPython] model_map: "+k+", "+v);
        });
        // [runPython] model_map: variant_model, minecraft:block/azalea
        // [runPython] model_map: mod_jar, C:\Users\statiic\AppData\Roaming\.minecraft\versions\1.21.11\1.21.11.jar
        // [runPython] model_map: model_mod_dir, minecraft
        // [runPython] model_map: variant_model_path, E:\Coding Projects\ProjectFiles\IdeaProjects\Minecraft MDL to Source MDL\assets\assets\minecraft\models\block\azalea.json
        // [runPython] model_map: blockstates_model, E:\Coding Projects\ProjectFiles\IdeaProjects\Minecraft MDL to Source MDL\assets\assets\minecraft\blockstates\azalea.json

        // path to blockstates.json

        String model = "";
        //if (model_map.containsKey("model_mode")) {
        if (model_map.get("model_mode").equals("geometry")) {
            //if (model_map.get("model_mode").equals("geometry")) {
            // modify for geometry model
            //currentModelJsons.add(model_map.get("blockstates_model"));
            // model variant eg: minecraft:anvil.json?
            // model_map.put("variant_model", variant_model); // selected file minus the path to it,
            // model_map.put("model_mode", "geometry");
            // model_map.put("is_geo", "true");
            // model_map.put("variant_model_path", String.valueOf(geo_model)); // full path to json_model, of what is selected
            args.put("is_geo", "True");
            args.put("geo_path", model_map.get("variant_model_path"));
            model = model_map.get("variant_model").replace(".json", "");
            args.put("mod_jar", "");
            currentModelJsons.add(model_map.get("variant_model_path"));

            //TODO some models may not come back and leave neither of these to set, making the logs off
            args.put("mc_jar", "");
            System.out.println("[runPython] model: " + model);
            //}
        } else {
            if (!sourcecraft_importing.toString().isEmpty()) {
                model = sourcecraft_importing.toString().replace(".json", "");
//                String mod_jar = model_map.get("mod_jar");
//                args.put("mod_jar", mod_jar);
            }
            else {
                currentModelJsons.add(model_map.get("blockstates_model"));
                model = model_map.get("variant_model").replace(".json", "");
//                String mod_jar = model_map.get("mod_jar");
//                args.put("mod_jar", mod_jar);
            }

            String mcjar = settings.get_name_in_jar_paths("minecraft");
            if (mcjar.isEmpty()) {
                logger("Minecraft jar must be added, click '+' and add jar path.", 0);
                return;
            }
            String mod_jar = model_map.get("mod_jar");
            args.put("mod_jar", mod_jar);

            //TODO some models may not come back and leave neither of these to set, making the logs off

            args.put("mc_jar", mcjar);
            args.put("is_geo", "False");
            args.put("geo_path", "''");
            System.out.println("[runPython] model: "+ model);
        }

        PythonRunner pr = new PythonRunner();
        try {
            logsLogger(pr.PythonRun(model, args));

        } catch (IOException | InterruptedException ex) {
            throw new RuntimeException(ex);
        }


        String[] log_lines = build_logs.getLast().split("\n");
        boolean completed = false;
        //boolean line_logged = false;

        for (String line : log_lines) {
            if (line.startsWith("Completed") && line.endsWith(".qc\"")) {
                logger("--> " + line.replace(".qc", ".json"), 0);
                completed = true;
            }
        }
        if (!completed) {
            logger("--> " + model + ".json failed.", 0);
            String item = " ❌ " + model;
            model_log_selector_cb.addItem(item);
            model_log_selector_cb.setSelectedItem(item);
        }
        if (completed) {
            String item = " ✔️ " + model;
            model_log_selector_cb.addItem(item);
            model_log_selector_cb.setSelectedItem(item);
        }

    }


    /// if model is blockState file, convert variant models
    /// @param model path to blockstate_name.json from assets/ of jar
    /// @return map of properties of model
    public HashMap<String, HashMap<String, String>> doBlockStates(File model, String block) throws IOException {
        // model is item_value from what was selected in model list, its file path, relative to assets/
        File model_path = new File(model.getPath());
        System.out.println("model_path: " + model_path);
        // name from jar?
        // will need to go with full path
        File mod_jars;
        String jar_path;
        if (model_path.toString().startsWith("assets\\")) {
            String mod_name = jars_list.getSelectedValue().toString();
//            if (mod_name.contains("/")) {
//                mod_name = mod_name.split("/")[0];
//                // will need the other part for what folder to look in
//            }
            System.out.println("mod_name: "+ mod_name);
            //modListMaps.forEach((k,v) -> System.out.println(k+", "+v));
            jar_path = modListMaps.get(mod_name).toString();
        }
        else {
            //String model_path_temp = model_path.toString().split("assets\\\\assets\\\\")[1] + "assets\\assets\\";
            String model_path_temp = model_path.toString().substring(model_path.toString().lastIndexOf("assets"));
            model_path = new File(model_path_temp);
            System.out.println("model_path_temp: "+ model_path_temp);
            System.out.println("model_path: " + model_path);
            mod_jars = new File(sourcecraft_importing.getParent(), "mod_jars.json");
            if (!block.split(":")[0].equals("minecraft"))
                jar_path = getJsonContent(mod_jars).get(block);
            else
                jar_path = settings.get_name_in_jar_paths("minecraft");
        }

        // need to check for if a blockstate or model now
        // model should be model.json minus bs: or model: prefix
        // should this even run for a model: prefix?

        String fileToExtract = model_path.toString().replace("\\", "/");
        String destDir = model_path.toString().split("assets")[0] + "assets";
        System.out.println("fileToExtract: " + fileToExtract + ", from jar_path: " + jar_path + ", to destDir: " + destDir);
        assets_dir = new File(fileToExtract).getParent();
        String extracted_path = extractFileFromJar(jar_path, fileToExtract, destDir);
        System.out.println("extracted_path: "+ extracted_path);
        assert extracted_path != null;
        model_path = new File(extracted_path);
        //currentModelJsons.add(model_path.toString());

        Gson gson = new GsonBuilder().create();
        if (model_path.exists()) {
            System.out.println("getting fileAsString...");
            String fileAsString = new String(Files.readAllBytes(model_path.toPath()));
            System.out.println("fileAsString.length(): " + fileAsString.length());
            BLOCKSTATES = gson.fromJson(fileAsString, BlockStates.class);

            System.out.println("BLOCKSTATES.variants: " + BLOCKSTATES.variants);
            System.out.println("Keyset?: " + BLOCKSTATES.variants.keySet());
            BLOCKSTATES.getUniqueModels();
            //ArrayList<String> models = new ArrayList<>();
            HashMap<String, HashMap<String, String>> models_map = new HashMap<>();

            //TODO clean up hack job
            for (String mdl : BLOCKSTATES.models) {
                // take mdl eg minecraft:anvil and separate the mod from the model path
                // add to array of variant models of block state
                String path_to_assets = model_path.getParent().replace("blockstates", "models");
                //String path_to_assets = model_path.toString().substring(0, model_path.toString().indexOf("/blockstates"));
                //String name = mdl.split(":")[1].replace("block", "") + ".json";
                String name = mdl.split(":")[1] + ".json";
                String new_path = new File(path_to_assets, name).toString();
                System.out.println("new_path: " + new_path);
                System.out.println("BLOCKSTATES.model mdl: " + mdl);
                //models.add(new_path);
                String file_to_extract = new_path.substring(new_path.indexOf("assets")+7);
                String extracted_path2 = extractFileFromJar(jar_path, file_to_extract, destDir);

                // should probably return model game name, model path to block_states model, model mod to fill in for args going to python
                // maybe should put all values into BLOCKSTATES?
                HashMap<String, String> model_map = new HashMap<>();
                model_map.put("variant_model", mdl);
                model_map.put("variant_model_path", extracted_path2);
                model_map.put("model_mod_dir", mdl.split(":")[0]);
                model_map.put("mod_jar", jar_path);
                model_map.put("blockstates_model", model_path.toString());
                model_map.put("model_mode", "java_model");
                models_map.put(mdl, model_map);
            }
            // model_path is full absolute path to model
            // need to return full path to variant models (C://.../assets/assets/minecraft/models/block/model_name)
            // normally this will return the variants of a blockstate selected
            // then bs_arr is populated with the variant files (minecraft:model) format
            return models_map;
            //return models;
            //return BLOCKSTATES.models;
        }

        return null;
    }
    public HashMap<String, HashMap<String, String>> doModelList(File model) throws IOException {
        // model is item_value from what was selected in model list, its file path, relative to assets/
        String block = "something:model";
        File model_path = new File(model.getPath());
        String mod_name = block.split(":")[0];
        System.out.println("model_path: " + model_path);
        // name from jar?
        // will need to go with full path
        if (model_path.toString().startsWith("assets\\")) {
            //String mod_name = jars_list.getSelectedValue().toString();
            //if (mod_name.contains("/")) {
            //    mod_name = mod_name.split("/")[0];
            //    // will need the other part for what folder to look in
            //}
            // need to get jar not from whats imported in the mods list but from the mod the file comes from
            // we wont know what that is if its imported from sourcecraft
            // either pass that thruogh the imports script or obtain it through searching here
            //String jar_path = modListMaps.get(mod_name).toString();
            File mod_jars = new File(sourcecraft_importing.getParent(), "mod_jars.json");
            String jar_path = getJsonContent(mod_jars).get(block);

            // need to check for if a blockstate or model now
            // model should be model.json minus bs: or model: prefix
            // should this even run for a model: prefix?

            String fileToExtract = model_path.toString().replace("\\", "/");
            String destDir = model_path.toString().split("assets")[0] + "assets";
            System.out.println("fileToExtract: " + fileToExtract + ", from jar_path: " + jar_path + ", to destDir: " + destDir);
            assets_dir = new File(fileToExtract).getParent();
            String extracted_path = extractFileFromJar(jar_path, fileToExtract, destDir);
            assert extracted_path != null;
            model_path = new File(extracted_path);
            //currentModelJsons.add(model_path.toString());

            Gson gson = new GsonBuilder().create();
            if (model_path.exists()) {
                System.out.println("getting fileAsString...");
                String fileAsString = new String(Files.readAllBytes(model_path.toPath()));
                System.out.println("fileAsString.length(): " + fileAsString.length());
                BLOCKSTATES = gson.fromJson(fileAsString, BlockStates.class);

                System.out.println("BLOCKSTATES.variants: " + BLOCKSTATES.variants);
                //System.out.println("Keyset?: " + BLOCKSTATES.variants.keySet());
                BLOCKSTATES.getUniqueModels();
                ArrayList<String> models = new ArrayList<>();
                HashMap<String, HashMap<String, String>> models_map = new HashMap<>();

                //TODO clean up hack job
                for (String mdl : BLOCKSTATES.models) {
                    // take mdl eg minecraft:anvil and separate the mod from the model path
                    // add to array of variant models of block state
                    String path_to_assets = model_path.getParent().replace("blockstates", "models");
                    //String path_to_assets = model_path.
                    //String path_to_assets = model_path.toString().substring(0, model_path.toString().indexOf("/blockstates"));
                    String name = mdl.split(":")[1].replace("block", "") + ".json";
                    //String name = block.split(":")[1] + ".json";
                    String new_path = new File(path_to_assets, name).toString();
                    System.out.println("new_path: " + new_path);
                    System.out.println("BLOCKSTATES.model mdl: " + block);
                    //models.add(new_path);
                    //String file_to_extract = new_path.substring(new_path.indexOf("assets")+7);

                    String extracted_path2 = extractFileFromJar(jar_path, model_path.toString(), destDir);
                    HashMap<String, String> model_map = new HashMap<>();
                    model_map.put("variant_model", block);
                    model_map.put("variant_model_path", extracted_path2);
                    model_map.put("model_mod_dir", block.split(":")[0]);
                    model_map.put("mod_jar", jar_path);
                    model_map.put("blockstates_model", "");

                    models_map.put(block, model_map);
                }
                // model_path is full absolute path to model
                // need to return full path to variant models (C://.../assets/assets/minecraft/models/block/model_name)
                // normally this will return the variants of a blockstate selected
                // then bs_arr is populated with the variant files (minecraft:model) format
                return models_map;
                //return models;
                //return BLOCKSTATES.models;
            }
            else System.out.println("model_path.exists() is false. model_path: "+ model_path);
        }
        return null;
    }
    public File[] fileChooserDialog(boolean is_file) {
        JFileChooser file_chooser = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
        FileFilter filter = new FileFilter() {
            @Override
            public boolean accept(File f) {
                //if (f.getName().equals("chest")) return false;
                if (f.getName().startsWith("template_")) return false;
                if (f.getName().contains(".json")) return true;
                if (f.isDirectory()) return true;
                return false;
            }
            @Override
            public String getDescription() {
                return ".json files";
            }
        };
        if (is_file) {
            // when selecting json models
            //System.out.println("has_key: " + settings.has_key("file_chooser_dir"));
            if (settings.has_key("file_chooser_dir"))
                file_chooser_dir = new File(settings.get_file_chooser_dir());
            else file_chooser_dir = new File(UserDir + "/assets");

            file_chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
            file_chooser.setMultiSelectionEnabled(true);
            file_chooser.setFileFilter(filter);
            file_chooser.setCurrentDirectory(file_chooser_dir);
            file_chooser.setSize(600, 800);
            file_chooser.setPreferredSize((new Dimension(600, 800)));

            int returnVal = file_chooser.showOpenDialog(null);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                file_chooser_dir = file_chooser.getCurrentDirectory();
                settings.set_file_chooser_dir(file_chooser_dir.toString());
                // TODO: get actual assets folder, currently assets must be from root/assets/assets folder only
                //String path_to = file_chooser_dir.getPath().replace(UserDir, "");
                //String path_to = file_chooser_dir.getPath();
                assets_dir = file_chooser_dir.getPath();
                System.out.println("file_chooser.getSelectedFiles(): " + file_chooser.getSelectedFiles());
                return file_chooser.getSelectedFiles();
            }
        }
        else {
            // when selecting steamapps path
            file_chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            file_chooser.setMultiSelectionEnabled(true);
            //file_chooser.setCurrentDirectory(file_chooser_dir);
            int returnVal = file_chooser.showOpenDialog(null);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                file_chooser_dir = file_chooser.getCurrentDirectory();
                return file_chooser.getSelectedFiles();
            }
        }
        return null;
    }
    public ArrayList<File> JarAdderDialog() {
        JFileChooser jar_chooser = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
        FileFilter filter = new FileFilter() {
            @Override
            public boolean accept(File f) {
                if (f.getName().contains(".jar")) return true;
                if (f.isDirectory()) return true;
                return false;
            }
            @Override
            public String getDescription() {
                return ".jar files";
            }
        };

        String userHome = System.getProperty("user.home");
        jar_chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        jar_chooser.setMultiSelectionEnabled(true);
        jar_chooser.setFileFilter(filter);
        jar_chooser.setCurrentDirectory(jar_adder_dir); //TODO
        jar_chooser.setSize(600, 800);
        jar_chooser.setPreferredSize((new Dimension(600, 600)));

        int returnVal = jar_chooser.showOpenDialog(null);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            jar_adder_dir = jar_chooser.getCurrentDirectory();
            settings.set_jar_chooser_dir(jar_adder_dir.toString());
            // TODO: get actual assets folder, currently assets must be from root/assets/assets folder only
            //String path_to = file_chooser_dir.getPath().replace(UserDir, "");
            //String path_to = file_chooser_dir.getPath();
            //assets_dir = jar_chooser_dir.getPath();

            return new ArrayList<>(Arrays.asList(jar_chooser.getSelectedFiles()));
        }
        return null;
    }
    /// logger( str: String, priority: int (0 = status, 1 = console, 2 = build, 3 = everything)
    public void logger(String str, int priority) {
        if (priority == 0) {
            status_log += str + "\n";
            status_output.setText(status_log);
            System.out.println(str);
        }
        if (priority == 1) {
            console_log += str + "\n";
            console_output.setText(console_log);
            System.out.println(str);
        }
        if (priority == 2) {
            build_log += str + "\n";
            build_output.setText(build_log);
            System.out.println(str);
        }
    }
    /// log to console/debug logs from mcexport.py without printing to system out
    /// Parameters: logs: String[]
    public void logsLogger(String[] logs) {
        //logger(logs[0], 0);
        status_log += logs[0];
        status_output.setText(status_log);

        //logger(logs[1], 1);
        console_logs.add(logs[1]);
        //console_log += logs[1];
        console_output.setText(console_logs.getFirst());

        build_logs.add(logs[2]);
        //build_log = logs[2];
        build_output.setText(build_logs.getFirst());

        // print full results
        System.out.println(logs[3]);

        setModelJSONS(logs[4]);
    }
    /// set JSON Model files to array for opening in notepad.exe
    private void setModelJSONS(String logs) {

        String[] paths = logs.split("\n");
        System.out.println("paths: ");
        for (String path : paths) {
            //currentModelJsons.add(path);
            currentModelJsons.add(path);
            System.out.println("path: "+path);
        }
        //String[] arr = paths.length
        //modelJsons.add(currentModelJsons);
        modelJsons.add(new ArrayList<>(currentModelJsons));
        currentModelJsons.clear();
    }
    /// get jsonPath content back as Map<String, Object>
    public Map<String, String> getJsonContent(File jsonPath) {
        Gson gson = new GsonBuilder().create();
        String fileAsString;
        try {
            //System.out.println(jsonPath.toPath());
            fileAsString = new String(Files.readAllBytes(jsonPath.toPath()));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
        Type type = new TypeToken<Map<String, String>>(){}.getType();
        return gson.fromJson(fileAsString, type);
    }
}
