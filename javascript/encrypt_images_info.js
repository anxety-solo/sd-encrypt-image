lsLoad = false;
onUiUpdate(function () {
  if (lsLoad) return;
  if (!opts) return;
  if (!opts["encrypt_image_is_enable"]) return;
  lsLoad = true;
  let enable = opts["encrypt_image_is_enable"] == "Yes";
});