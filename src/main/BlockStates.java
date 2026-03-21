package main;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class BlockStates {

	public Map<String, Map<String, String>> variants;
    public ArrayList<Map<String, Map<String, String>>> multipart;
    public String model;
    public ArrayList<String> models = new ArrayList<>();

    public int x = 0;
    public int y = 0;

	public BlockStates() {

        this.variants = new HashMap<>();
        this.multipart = new ArrayList<>();
        System.out.println("blockstates initiated.");
	}

    public void getUniqueModels() {
        for ( Map.Entry<String, Map<String, String>> keys : this.variants.entrySet() ) {
            if (!check_in_array(keys.getValue().get("model")))
                this.models.add(keys.getValue().get("model"));
        }
        System.out.println("multipart.size: " + multipart.size());

        for (Map<String, Map<String, String>> part : this.multipart) {

            System.out.println("part.size: " + part.size());
            System.out.println("part.entrySet(): " + part.entrySet());
            for ( Map.Entry<String, Map<String, String>> keys : part.entrySet() ) {
                System.out.println("keys: " + keys);
                System.out.println("keys.getValue().get(model): " + keys.getValue().get("model"));

                System.out.println("keys.getKey(): " + keys.getKey());
                if (keys.getValue().get("model") != null) {
                    if (!check_in_array(keys.getValue().get("model")))
                        this.models.add(keys.getValue().get("model"));
                }
            }
        }
        System.out.println("does it get here?");
        System.out.println("this.models: " + this.models);

        //this.models.forEach(m -> System.out.println("model: " + m));
    }
    private boolean check_in_array(String m) {
        for (String model : this.models) {
            // check in array for model already added
            if (m.equals(model)) {
                return true;
            }
        }
        return false;
    }
}
