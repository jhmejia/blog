import lume from "lume/mod.ts";
import picture from "lume/plugins/picture.ts";
import transformImages from "lume/plugins/transform_images.ts";
import plugins from "./plugins.ts";

const site = lume({
  src: "./src",
});

site.use(plugins());
site.use(picture(/* Options */));
site.use(transformImages());

export default site;
