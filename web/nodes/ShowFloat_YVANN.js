import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
	name: "ShowFloat_YVANN.ShowFloat_YVANN",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ShowFloat_YVANN") {
			// When the node is executed we will be sent the input float, display this in the widget
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "float");
					if (pos !== -1) {
						for (let i = pos; i < this.widgets.length; i++) {
							this.widgets[i].onRemove?.();
						}
						this.widgets.length = pos;
					}
				}

				for (const value of message.float) {
					const w = ComfyWidgets["FLOAT"](this, "float", ["FLOAT", { precision: 4 }], app).widget;
					w.inputEl.readOnly = true;
					w.inputEl.style.opacity = 0.6;
					w.value = value;
				}

				this.onResize?.(this.size);
			};
		}
	},
});