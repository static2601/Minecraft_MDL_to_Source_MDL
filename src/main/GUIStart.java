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
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

import static main.JarExtractor.*;

public class GUIStart {
    public static getSettingsData settings;
    private JPanel jPanel1;
    private JPanel main_jPanel;
    //private JList jars_list;
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
    //private JTextField mcjar_textField;
    //private JButton mcjar_search_button;
    private JPanel console_jpanel;
    private JTabbedPane tabbedPane2;
    private JList jars_list;
    private JList jar_files_list;
    private JButton jar_add_button;
    private JButton jar_sort_button;
    private JButton jar_remove_button;
    private JLabel availableModsLabel;
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
    private HashMap<String, String> modFileListMaps = new HashMap<>();


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
        //model_files = new ArrayList<>();
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
                // convert paths for tools and game folders
                if (!jar_files_list.getSelectedValuesList().isEmpty()) {
                    List<String> selected_items = jar_files_list.getSelectedValuesList();
                    model_files.clear();

                    for (String item : selected_items) {
                        File item_value = new File(modFileListMaps.get(item));

                        try {
                            HashMap<String, HashMap<String, String>> bs_maps = doBlockStates(item_value);
                            for (String key : bs_maps.keySet()) {
                                HashMap<String, String> model = bs_maps.get(key);
                                System.out.println("adding model: "+ model.get("variant_model") + " to model_files");
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

                        for (HashMap<String, String> mdl : model_files) {
                            System.out.println("mdl: "+ mdl);
                            runPython(mdl, args);
                        }
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
                            ProcessBuilder pb = new ProcessBuilder("notepad.exe", model_path);
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
                    ArrayList<String> filePaths = modelJsons.get(selectedIndex);
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
                System.out.println("getValueIsAdjusting: " + jars_list.getSelectedValue());
                // with mod name selected get blockstates of mod

                String mod = jars_list.getSelectedValue().toString();
                File jar_path = modListMaps.get(mod);
                ArrayList<String> modFiles = getBlockStatesOfMod(mod, jar_path);
                //System.out.println(modFiles.toString());
                //if (modFiles == null) return;
                listModel2.removeAllElements();
                modFileListMaps.clear();
                for (String f : modFiles) {
                    // get only the model.json
                    String json = f.substring(f.lastIndexOf("/")+1);
                    String json_name = json.replace(".json", "");
                    // needs to extract files to a temp folder to be converted
                    // or get a copy of the json and process, but would need to change how it converts model_files
                    modFileListMaps.put(json_name, f);
                    //System.out.println("f: "+ f);
                    listModel2.addElement(json_name);
                }
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
    }

    private void setToModList(ArrayList<File> selectedFiles) {
        // add jar file
        //ArrayList<File> selectedFiles = JarAdderDialog();
        if (selectedFiles == null) return;

        for (File file : selectedFiles) {
            // f is .jar
            //settings.set_jar_paths(f.toString());
            //TODO make single statement
            if (file.isDirectory()) {
                for (File f : Objects.requireNonNull(file.listFiles())) {

                    HashMap<String, File> map = getModList(f);
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
            }
            else {

                HashMap<String, File> map = getModList(file);
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

        // path to blockstates.json
        currentModelJsons.add(model_map.get("blockstates_model"));
        String mcjar = settings.get_name_in_jar_paths("minecraft");
        if (mcjar.isEmpty()) {
            logger("Minecraft jar must be added, click '+' and add jar path.",0);
            return;
        }
        args.put("mc_jar", mcjar);

        String model = model_map.get("variant_model").replace(".json", "");
        String mod_jar = model_map.get("mod_jar");
        args.put("mod_jar", mod_jar);

        System.out.println("[runPython] model: "+ model);

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
    /// @return map of properties of model
    public HashMap<String, HashMap<String, String>> doBlockStates(File model) throws IOException {
        // model is item_value from what was selected in model list, its file path, relative to assets/
        File model_path = new File(model.getPath());
        System.out.println("model_path: " + model_path);
        if (model_path.toString().startsWith("assets\\")) {
            String mod_name = jars_list.getSelectedValue().toString();
            if (mod_name.contains("/")) {
                mod_name = mod_name.split("/")[0];
                // will need the other part for what folder to look in
            }

            String jar_path = modListMaps.get(mod_name).toString();

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
                System.out.println("Keyset?: " + BLOCKSTATES.variants.keySet());
                BLOCKSTATES.getUniqueModels();
                ArrayList<String> models = new ArrayList<>();
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
                    models.add(new_path);
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
            System.out.println("has_key: " + settings.has_key("file_chooser_dir"));
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
}
