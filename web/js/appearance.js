import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Yvann.appearance", // Extension name
    async nodeCreated(node) {
        if (node.comfyClass.endsWith("Yvann")) {
            // Apply styling
            node.color = "#153a61";
            node.bgcolor = "#1A4870";


        }
    }
});

//#51829B bleu fonce