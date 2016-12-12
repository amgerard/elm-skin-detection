# elm-skin-detection
Using SLIC SuperPixels and Extreme Learning Machines (ELM) for Skin Detection.

URL:
http://192.241.144.178/bd/

Testing Model with Test Dataset:

The code is located in the following folder on the cluster:

/Shared/bdagroup3/Fall2016/elm-skin-detection

Which contains a bash script ‘run_test.sh’ that will setup the environment, load the model and test data, run the test and print the results. Make sure to run using the source command or . command:

    . /Shared/bdagroup3/Fall2016/elm-skin-detection/run_test.sh

NOTE

The path to the test images and test masks are hardcoded in the scripts to:

    /Shared/bdagroup3/Original/test

And

    /Shared/bdagroup3/Skin_test

I still can’t access the Skin_test directory, so I’m not sure if this path is correct, or if the masks are in a subfolder of that directory. You can pass the paths to the test images and test masks into run_test.sh like so:

    . run_test.sh /Shared/bdagroup3/Original/test /Shared/bdagroup3/Skin_test
