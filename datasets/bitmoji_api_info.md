# Bitmoji API Guide #

### General Command ###

```
https://render.bitstrips.com/render/%e/%u-v1.png?pd2={"%edit_prop_1":"%edit_arg_1","%edit_prop_2":"%edit_arg_2"}&%other_args
```
The "%" denotes where an argument shouuld be inserted, you should not actually include it in the command.

You must insert the following arguments:
* %e (expression/pose id): I believe this controls the expression of the emoji or the pose. **The only one that matters for our use is 6688424** as it's the default avatar editor pose.
* %u (user id): controls the base emoji that later arguments will edit. Mine is 371434407_3_s1.
    * As far as I can tell the 3 in "_3_" controls which saved iteration of the emoji to use.
    * The "s1" controls the style, larger number (e.g. "s5") can render the bitmoji style instead of bitstrips.
    * We should probably use one boy and one girl user id because the gender is not an editable property.
* %edit_prop (properties to edit the emoji): there are a number of properties that can be changed, see table below.
* %edit_arg (argument to set property): values that the properties can take, see table below.
* %other_args
  * This includes:
      * body=
      * cropped="head"
      * head_rotation=0
      * scale=2
      * style=1
  * I'm sure there are others, but these seem like the relevant ones that show up during the avatar editor.
  
  ### PD2 Properties and Values ###
  
| **Property**        | **Values**           | **Description**           |
| ------------- |:-------------:|:-------------:|
| col 3 is      | right-aligned | right-aligned |
| col 2 is      | centered      |  centered      |
| zebra stripes | are neat      | are neat      |

### Other Properties and Values

| **Property**        | **Values**           | **Description**           |
| ------------- |:-------------:|:-------------:|
| proportion      | 0-8 | face shape |
| colours      | {"ffcc99":16443344,"ff9866":16041410}, {"ffcc99":15257000,"ff9866":14530688}, {"ffcc99":11897407,"ff9866":11235404}, {"ffcc99":8080170,"ff9866":7549993}, {"ffcc99":16764057,"ff9866":16750694}, {"ffcc99":14664067,"ff9866":14199915}, {"ffcc99":12159077,"ff9866":11558746}, {"ffcc99":6963494,"ff9866":6173474}, {"ffcc99":16691590,"ff9866":14128499}, {"ffcc99":13544297,"ff9866":13076077}, {"ffcc99":11170379,"ff9866":10506048}, {"ffcc99":6240025,"ff9866":5908506}, {"ffcc99":12684916,"ff9866":11824997}, {"ffcc99":13280865,"ff9866":11242835}, {"ffcc99":9657655,"ff9866":8997431}, {"ffcc99":4732712,"ff9866":4138525} |   can be any combination of hex colors (w/ some weird id next to it, defaults are listed  |
| hair color | {"926715":8672042,"6f4b4b":6700322}, {"926715":6632737,"6f4b4b":5844766}, {"926715":4795690,"6f4b4b":3218460}, {"926715":2566954,"6f4b4b":1579802}, {"926715":14797722,"6f4b4b":11569973}, {"926715":12360500,"6f4b4b":9663272}, {"926715":22926715,"6f4b4b":7162651}, {"926715":5587258,"6f4b4b":3615014}, {"926715":16750848,"6f4b4b":14386178}, {"926715":14178816,"6f4b4b":11618049}, {"926715":11093553,"6f4b4b":8203556}, {"926715":10027008,"6f4b4b":7733505}, {"926715":16250871,"6f4b4b":15132390}, {"926715":13618371,"6f4b4b":7696224}, {"926715":10725013,"6f4b4b":9343614}, {"926715":8291180,"6f4b4b":3553071}     | add these args to the colour list     |
| eyebrow color |    {"4f453e":6700322}, {"4f453e":5844766}, {"4f453e":3218460}, {"4f453e":1579802}, {"4f453e":11569973}, {"4f453e":9663272}, {"4f453e":7162651}, {"4f453e":3615014}, {"4f453e":14386178}, {"4f453e":11618049}, {"4f453e":8203556}, {"4f453e":7733505}, {"4f453e":15132390}, {"4f453e":7696224}, {"4f453e":9343614}, {"4f453e":3553071}     | add this arg to the colour list       |
| beard color | {"6f4b4b":6700322} | add these args to the colour list |
| blush | {"ff9999":16624025} | add these args to the colour list |
| lipstick | {"ff9866":13442115} | add these args to the colour list |
