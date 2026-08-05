import qupath.lib.images.servers.LabeledImageServer

// Exports a grayscale uint8 mask PNG with the class IDs required by the
// spheroid-seg data pipeline (design doc section 2.2):
//   0 = background, 1 = loose cell, 2 = spheroid, 3 = organoid
//
// Usage: Automate -> Show script editor -> paste -> Run (or Run for project).
// Class names below must match EXACTLY the PathClasses assigned to the
// annotations (case-sensitive). Adjust here if annotators used other names.
// Requires QuPath >= 0.4.
//
// Annotation gotchas (learned the hard way):
// - Use area tools (ellipse / polygon / brush). The points tool creates
//   polylines with no area, which rasterize as near-zero pixels.
// - Every annotation MUST have its class assigned (right-click -> Set class),
//   otherwise it is rasterized as background.
// - Exported PNGs may carry a color palette; convert to plain grayscale
//   (e.g. with PIL) if the QC reports a 3-channel mask.
// - Always run qc.py before copying masks into data/masks/.

def imageData = getCurrentImageData()
def server = imageData.getServer()

// Output folder: <project>/masks if a project is open;
// otherwise a "masks" folder next to the source image
def outDir
if (getProject() != null) {
    outDir = buildFilePath(PROJECT_BASE_DIR, "masks")
} else {
    def imgPath = new File(server.getURIs().find { it.getScheme() == "file" }.getPath())
    outDir = new File(imgPath.getParent(), "masks").getAbsolutePath()
    print "No project open: saving next to the image in " + outDir
}
mkdirs(outDir)

// Same base name as the source image (spec: data/masks/<base>.png)
def name = GeneralTools.stripExtension(server.getMetadata().getName())
def path = buildFilePath(outDir, name + ".png")

// 1 = full resolution. The spec requires the mask to have the exact same
// dimensions as the source image. Do not increase.
double downsample = 1

def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0)          // 0 = background (everything not annotated)
    .downsample(downsample)
    .addLabel("Loose cell", 1)   // 1 = loose cell
    .addLabel("Spheroid", 2)     // 2 = spheroid
    .addLabel("Organoid", 3)     // 3 = organoid
    .multichannelOutput(false)   // single-channel grayscale, one value per pixel
    .grayscale()                 // grayscale LUT (not per-class colors)
    .build()

writeImage(labelServer, path)
print "Mask exported: " + path
