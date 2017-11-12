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
