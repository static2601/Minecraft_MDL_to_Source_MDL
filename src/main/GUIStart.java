package main;

import org.json.simple.JSONObject;
import org.json.simple.parser.ParseException;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class GUIStart {
    private getSettingsData settings;
    private JPanel jPanel1;
    private JPanel main_jPanel;
    private JList list1;
    private JButton model_selection_btn;
    private JTextField model_txtfield;
    private JLabel assetsFolderLabel;
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
    private File file_chooser_dir;
    private final String UserDir;
    public String output_log;
    public String status_log;
    private ArrayList<File> model_files = new ArrayList<>();
    //private ArrayList<File> blacklisted_files = new ArrayList<>();

    static void main(String[] args) throws IOException, ParseException {
        try {
            for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (Exception e) {
            // If Nimbus is not available, fall back to cross-platform
            try {
                UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
            } catch (Exception ex) {}
        }
        new GUIStart();
    }
    public GUIStart() throws IOException, ParseException {

        settings = new getSettingsData();
        // set this script as a reference in getSettingsData
        // so we can call back to here non-staticly
        settings.guiStartRef = this;

        JFrame frame = new JFrame();
        frame.setContentPane(jPanel1);
        frame.setTitle("MC Model to Source MDL");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        //blacklisted_files.add("");

        status_log = "";
        output_log = "";
        UserDir = System.getProperty("user.dir");
        model_scale_spinner.setValue(48);
        skybox_scale_spinner.setValue(16);

        // reload saved data
        System.out.println("settings.settings.tostring: "+ settings.settings);
       try{
            steamapps_txtfield.setText(settings.get_steamappsPath());
            mod_combobox.setSelectedItem(settings.get_modName());
            game_combobox.setSelectedItem(settings.get_shortGame());
            model_scale_spinner.setValue(settings.get_scale());
            skybox_scale_spinner.setValue(settings.get_skybox_scale());
        } catch (Exception e) {
           System.out.println(e);
       }


        // listeners
        model_selection_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                File[] selectedFiles = fileChooserDialog(true);

                if (selectedFiles != null) {
                    model_files.clear();
                    for (File selectedFile : selectedFiles) {
                        model_files.add(selectedFile);
                    }
                    StringBuilder files_output = new StringBuilder();
                    String c = ", ";
                    for (File model_file : model_files){
                        files_output.append(model_file.getName().replace(".json", "") + c);
                    }
                    String names = "";
                    if (files_output.toString().endsWith(c)) {
                        names = files_output.toString();
                        names = names.substring(0, names.length()-2);
                    }
                    logger(names, 0);
                    model_txtfield.setText(names);
                    //settings.set_settingsKey("mcJarPath", selectedFile.getAbsolutePath());
                }
            }
        });
        // convert button - start conversion
        convert_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                if (steamapps_txtfield.getText().isEmpty()) {
                    logger("Steamapps Path must be filled.", 0);
                    return;
                }
                if (model_txtfield.getText().isEmpty()) {
                    logger("JSON model must be selected.", 0);
                    return;
                }

                // convert paths for tools and game folders
                String game_path = "";
                String tools_path = "";
                String steamapps_path = new File(steamapps_txtfield.getText()).getPath();
                String game_selection = game_combobox.getSelectedItem().toString().toLowerCase();

                if (game_selection.equals("tf2")) {
                    game_path = steamapps_path + "\\common\\Team Fortress 2\\tf";
                    tools_path = steamapps_path + "\\common\\Team Fortress 2\\bin";
                }else if (game_selection.equals("gmod")) {
                    game_path = steamapps_path + "\\common\\GarrysMod\\garrysmod";
                    tools_path = steamapps_path + "\\common\\GarrysMod\\bin";
                }
                logger("game_path: " + game_path, 0);
                logger("tools_path: " + tools_path, 0);

                // save data to json
                settings.set_scale( (Integer) model_scale_spinner.getValue());
                settings.set_skybox_scale( (Integer) skybox_scale_spinner.getValue());
                settings.set_shortGame(game_combobox.getSelectedItem().toString());
                settings.set_modName(mod_combobox.getSelectedItem().toString());
                settings.writeJson();

                // run pythonrunner
                Map<String, String> args = new HashMap<>();
                args.put("modname", mod_combobox.getSelectedItem().toString().toLowerCase());
                args.put("tools", tools_path);
                args.put("game", game_path);
                args.put("scale", model_scale_spinner.getValue().toString());
                args.put("skybox_scale", skybox_scale_spinner.getValue().toString());

                // if true, tell mcexports.py to use skybox scale instead of regular
                if (make_skybox_models_checkbox.isSelected())
                    args.put("compile_skybox", "true");
                else args.put("compile_skybox", "false");

                for (File model : model_files) {
                    String model_file = model.getName().replace(".json", "");
                    logger(model_file, 0);

                    PythonRunner pr = new PythonRunner();
                    try {
                        logger(pr.PythonRun(model_file, args)[1], 1);
                    } catch (IOException | InterruptedException ex) {
                        throw new RuntimeException(ex);
                    }
                }
                logger("model_files:", 0);
                model_files.forEach((n) -> logger(n.toString(), 0));

            }
        });
        steamapps_dir_search_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //logger(e.toString(), 0);
                File[] selectedFile = fileChooserDialog(false);
                if (selectedFile != null) {
                    String s = selectedFile[0].toString();
                    logger(s, 0);
                    s = s.substring(0, s.indexOf("steamapps")+9);

                    logger(s, 0);
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
                output_log = "";
                status_log = "";
            }
        });
        open_json_files_btn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent evt) {
                for (File model : model_files) {
                    String filePath = model.toString();
                    try {
                        ProcessBuilder pb = new ProcessBuilder("notepad.exe", filePath);
                        pb.start();
                    } catch (IOException e) {
                        System.err.println("An error occurred trying to open Notepad: " + e.getMessage());
                        e.printStackTrace();
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
                String filePath = "";
                String steamapps_path = new File(steamapps_txtfield.getText()).getPath();
                String game_selection = game_combobox.getSelectedItem().toString().toLowerCase();
                if (!steamapps_path.isEmpty()) {
                    if (game_selection.equals("tf2")) {
                        filePath = steamapps_path + "\\common\\Team Fortress 2\\tf";
                    }
                    else if (game_selection.equals("gmod")) {
                        filePath = steamapps_path + "\\common\\GarrysMod\\garrysmod";
                    }

                    try {
                        ProcessBuilder pb = new ProcessBuilder("explorer.exe", filePath);
                        pb.start();
                    } catch (IOException e) {
                        System.err.println("An error occurred trying to open File Explorer: " + e.getMessage());
                        e.printStackTrace();
                    }
                }
                else {
                    logger("Select 'Steamapps' first!", 0);
                }
            }
        });
    }

    public File[] fileChooserDialog(boolean is_file) {
        JFileChooser file_chooser = new JFileChooser();
        //file_chooser_dir = new File(UserDir + "/assets");


        //logger(file_chooser.toString(), 0);
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

            int returnVal = file_chooser.showOpenDialog(null);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                file_chooser_dir = file_chooser.getCurrentDirectory();
                settings.set_file_chooser_dir(file_chooser_dir.toString());
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
    /// logger( String str, int priority = (0 = print to overall, 1 = print to console) )
    public void logger(String str, int priority) {
        if (priority == 0) {
            status_log += str + "\n";
            status_output.setText(status_log);
            System.out.println(str);
        }
        if (priority == 1) {
            output_log += str + "\n";
            console_output.setText(output_log);
            System.out.println(str);
        }
    }
    /// called from getSettingsData() after a reference set for this script
//    public void setTexturePackCombo(JSONObject tp){
//        this.texturePacksJSON = tp;
//        Set keys = tp.keySet();
//        keys.forEach(k ->{
//            this.texturepack_comboBox.addItem(k);
//        });
//    }
}
