import { app } from "../../../scripts/app.js";

const COLOR_THEMES = {
    blue: { nodeColor: "#153a61", nodeBgColor: "#1A4870" },
};

const NODE_COLORS = {
    "Audio Analysis": "blue",
    "IPAdapter Audio Transitions": "blue",
    "Audio Prompt Schedule": "blue",
    "Audio Peaks Alternate": "blue",
    "AnimateDiff Audio Reactive": "blue",
    "ControlNet Audio Reactive": "blue",
    "Floats To Weights Strategy": "blue",
    "Invert Floats": "blue",
    "Floats Visualizer": "blue",
    "Mask To Float": "blue",
};

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];  // Swap elements
    }
}

let colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
shuffleArray(colorKeys);  // Shuffle the color themes initially

function setNodeColors(node, theme) {
    if (!theme) { return; }
    node.shape = "box";
    if (theme.nodeColor && theme.nodeBgColor) {
        node.color = theme.nodeColor;
        node.bgcolor = theme.nodeBgColor;
    }
}

const ext = {
    name: "Yvann.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];

            if (colorKey === "random") {
                // Check for a valid color key before popping
                if (colorKeys.length === 0 || !COLOR_THEMES[colorKeys[colorKeys.length - 1]]) {
                    colorKeys = Object.keys(COLOR_THEMES).filter(key => key !== "none");
                    shuffleArray(colorKeys);
                }
                colorKey = colorKeys.pop();
            }

            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);
