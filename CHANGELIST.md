### List of Changes (release alpha v0.6.2)
1. Improved the initial pause to ensure the game is paused at the very first frame of each game 
2. Improved game speed up. Now the physical simulation results are exactly the same at different speed up speed. 

### List of Changes (release alpha v0.6.1)
1. The game will be paused after loading
2. The camera now fully zooms out instantly at the beginning of the game

### List of Changes (release alpha v0.6.0)
1. 6 example phase 3 novelties released
    L36T10-14 no gravity
    L36T20-24 A radioactive area that will kill any bird that touches it
    L37T10 The magician acts as a transmuter who turns any touched object to a pig 
    L37T20 The butterfly adds health points to the pigs nearby 
    L38T10-14 A time triggered storm that applies force to the left of the flying birds
    L38T20-24 A long wood object that will destroy itself when a bird collides with any object

2. Added a new NoveltyHint Command (72) that returns different levels of novelty hints in JSON object. It only works for hint level 1 (hint levels shown below as a reference).

  Hint hierarchy:
  Type 0: level of novelty in novelty hierarchy
  Type 1: specify “thing” that changed (e.g., which object, action, relation)
  Type 2: specify which attribute of the “thing” has changed or what has changed (e.g., gravity)
  Type 3: what is the value of the changed attribute, how has it changed?

The symbolic groundtruth will be set to devmode where novel object is labeled as "novel_object"
The input and output format is in README.md

### List of Changes (release alpha v0.5.14)
1. Fixed an issue that may cause no batch ground truth to be received if a large tap time is given. 

### List of Changes (release alpha v0.5.13)
1. Add a new command GetInitialStateScreenShot (71) that will return the screenshot of the initial state of the current game. The input and output format are the same as the original DoScreenshot (11) command.
2. Fixed a bug that causes an infinite loop in large games
3. Removed the rotation effect of novelty L13T20
4. Fixed an tap time issue that freezes the communication between the agent and the game 
5. Now tracking the science birds game zips with Git Large File System as the cached images for the new command takes too much space 

### List of Changes (release alpha v0.5.12)
1. Fixed an issue that may cause NullReferenceException when the sling is not present in some rare cases.
2. Integrated competition novelties (L1T12, L2T14, L2T15) 

### List of Changes (release alpha v0.5.11)
1. In 'given detection' mode, the labels for novel objects in the groundtruth will be novel_object. 
2. 6 phase 1 evaluation novelties are released as following:
    L11T50 Styroform Black (wood health x 4, mass reduce 90%)
    L11T130 Orange bird wood damage 0, all other parameters same as yellow birds
    L12T30 Yellow bird launch gravity increase to 0.7, i.e., the max reachable distance is shorter
    L12T110 Blue Bird Split to 5
    L13T20 Remove blue channel (the last two bits) of the colours of objects.
    L13T90 shift the coordinate origin to the center point of the slingshot
3. Removed outdated main function of ./src/demo/naive_agent_groundtruth.py 

### List of Changes (release alpha v0.5.10)
1. Solved an issue that will cause some tap to be ignored.
2. Added more detailed description about tap time unit in README.md

### List of Changes (release alpha v0.5.9)
1. Improve the zoom out process. The game will be initialized at fully zoomed out state now. 

### List of Changes (release alpha v0.5.8)
1. Fixed some published level 2 novelty colour map issues.
2. Fixed a bug: when the game speed is greater than 1, the destroyed objects will stay in gt and go through the ground for a few frames.

### List of Changes (release alpha v0.5.7)
1. Fixed a bug: sometimes the butterfly will hit the sling and push the bird on slingshot away.
### List of Changes (release alpha v0.5.6)
1. Add representative game samples located in "./Levels/novelty_level_0/type222-227", "./Levels/novelty_level_0/type232-237", "./Levels/novelty_level_0/type242-247" and "./Levels/novelty_level_0/type252-257".
2. The initial location of the external agents will be slightly adjusted if they are collided with something at the beginning of the game.  
### List of Changes (release alpha v0.5.5)
1. Fixed a bug: when setting batch_gt_option = 1, the ground truth shoot (38) sometimes returns wrong segment of the batch ground truths.
### List of Changes (release alpha v0.5.4)
1. Fixed a bug: the butterfly is not in the ground truth
2. Fixed a bug: the game hangs occasionally due to shooting when there is not bird left
### List of Changes (release alpha v0.5.3)
1. Fixed a bug: when some external agents are moving, the Won/Lost Banner will not show up
### List of Changes (release alpha v0.5.2)
1. Fixed a bug: sometimes duplicated data entry is recorded as "playing" in EvaluationData due to delayed game state update.
2. Fixed an issue that if the tap time is set to too long, the game will hang due to waiting.
3. Tied the tap time to simulation time rather than real time to avoid any randomness due to system delay.
4. Increased the max number of frames to be recorded from 200 to 300 
5. Added an optional parameter for request 70 Batch Groundtruth. The agent can set any number of frames (greater than 0 and less than 300) to be recorded. The default value is 300.  
6. Changed the witch and flying pig logic. Now they fly horizontally and change direction every 1000 physical updates or reach the boundary of the scene.
7. Added sample games with simplified structures in ./Levels/novelty_level_22-25 and ./Levels/novelty_level_0/type22-25.   
### List of Changes (release alpha v0.5.1)
1. Added one novelty example each for new novelty level 2-5
  - the novelty level index for new hierarchy is added by 20 to distinguish with the old one
  - the new novelties examples are in folder ./Levels/novelty_level_22-25  
  - Novelty Level 22 type 1 the pig with red beard that moves right to left to try to dodge the bird's direct hit 
  - Novelty Level 23 type 1 the normal pig that flies
  - Novelty Level 24 type 1 the pig with red beard that cannot be killed by the hit from the objects other than birds
  - Novelty Level 25 type 1 the normal pig that can stay *in* the platform 
2. Added a new request 70: BatchGroundTruth
  - It takes one integer parameter: the frequency, n, of the batch ground truth. The ground truth will be recorded every n frames.
  - The maximal number of ground truths recorded is 200
  - The recording will also stop if no object in the scene is moving
  - If the game is paused, it will be resumed at the beginning of the recording
  - The game will be paused after the recording is finished
3. Fixed a bug: the game will not be resumed after the mouse scroll operation or zoom in/out request  
4. The old novelty examples are added back
### List of Changes (release alpha v0.5.0)
1. Changed request 38, shoot and record batch ground as below:
  - added an optional parameter to let the TA2 agent specify how the batch ground truth can be collected. 0 is the default value.
    -- there are two options in this version: 1. record full ground truths (at most 200 frames); 2. only record the ground truths during the bird flying before the tap or the first collision between the bird and another object.
  - the game will be paused after the game scene is stable or 200 frames of ground truths have been recorded
  - the game will be resumed when receiving another shooting command
2. Reduced position noise level to at most +-1 pixel
3. Fixed the trajectory dots issue in the game and ground truth
4. The time interval between two consective ground truths in a batch is consistent now 
5. Level loading actions do not count as interactions any more
6. Added warning message in the game log when trying to shoot while game is not in playing state. There will be no batch ground truth returned in such situation.
7. Fixed bug with colour map noise. The noise to the colour percentage was not applied correctly. Now each 'percentage' of the colour will have a 1% noise.
8. Adapted naive agent to the new batch gt request
9. Updated the implementation 
10. Added four external agents:
  - Magicians who walk on the ground and avoid to touch any other objects
  - Wizards who fly in the air and avoid to touch any other objects
  - Worms who walk in the platform
  - Butterflies who fly along the convex hull of the structures 

- Note: 
  - you are not required to modify your agent if you do not intend to use the new optional parameter that allows to collect bird flying ground truths. <br/>
  - You can simply append a 4-byte integer with value 0 or 1 to apply the option.

### List of Changes (release alpha v0.4.2)
1. Fixed grayscale novelty (L3T6)
2. Added true novelty object ID section in the recorded Likelihood file
  - The 'true novel IDs' section records the novel object IDs in the current state
  - The 'complete true novel IDs' section records the novel object IDs in the game 
3. changed default resolution for head mode to 640x480, which matches the resolution in headless mode used in all evaluations
4. Fixed some null object id issue
 
### List of Changes (release alpha v0.4.1 patch)
1. Added a command argument for game_playing_interface.jar: --noisy-batch-gt
    - if the argument is used, the batch ground truth command will always return noisy grount truths regardless of the dev mode

### List of Changes (release alpha v0.4.1)
1. Updated game set with the same distribution as in the evaluation
2. Fixed python game playing agent bug when decoding return message from Request-Novelty-Information (69) command. Now it can decode negative number correctly.
3. Added batch ground truth visualizer: src/utils/gt_gif_show.py

### List of Changes (release alpha v0.4.0)
1. Remove noise of batch groundtruth in dev mode
2. Add two baseline agents (datalab and eagle wings) and readme to run them
3. Update readme file
4. Change the name of sciencebirds.log to sciencebirds_[port_number].log
5. Speed up animation
6. Make sure new training set state follows new trial state

### List of Changes (release alpha v0.3.8)
1. Change the implemtation of simulation speed up. Any speed of the simulation gives the same result.
    - The simulation speed is not capped in the new implementation
2. Update the noise model for the coordinates of the objects from uniform to Gaussian distribution
3. Fix the "BindException" issue in game playing interface.
    - The port in TIME_WAIT state can be connected
    - If the port is in use, another free port will be used

### List of Changes (release alpha v0.3.7)
1. Shooting related commdands changed
    - only release point is needed
2. ReportNoveltyLikelihood command is extended
    - novelty object IDs 
    - novelty level
    - novelty description 
3. After loading a new game instance, the first groun truth will be returned after the scene is fully zoomed out
4. Novelty level 3 type 7 is not working in 0.3.6. It has been fixed in this version 
5. Adapted DQ agent to the new interface
### List of Changes (release alpha v0.3.6)
1. Development mode is added
  - use --dev when running the game playing interface to access more information about the game objects
  - under dev mode, the type of the object in the ground truth is the true object type
  - under dev mode, agent command groundtruthwithscreenshot and groundtruthwithoutscreenshot will return non-noisy groundtruth
  - under dev mode, agent command ShootAndRecordGroundTruth will return non-noisy groundtruth
2. Coordination system of the agent and the science birds game is consistent
3. The log of the science birds game can be found in the same folder as the game executable, named sciencebirds.log
4. The ground truth json format has been changed to GeoJSON inspired format (the differences can be found in README.md). Details about the new format is here[link] (an example of the new format can be found here[link]). 
5. The published novelty level 3 type 7 has been fixed
6. Add command line argument --agent-port that specify the port for the agent to connect
7. The old argument --agent-start-port has been renamed to --game-start-port to prevent misunderstanding

### List of Changes (release alpha v0.3.5)
1. Add --headlesss command line argument for that gameplayinginterface.jar that run the system headless
2. Reduce noise level to up to 2 pixels for the objects in the noisy ground truth
3. Change the object IDs in the groundtruth to unique IDs for each object that can be used for object tracking
4. Change the vertices representation for objects in the ground truth to a list of contours. Each contour contains a list of vertices.
5. Speed up the ground truth generation from ~100 (80 headless) ms/frame to ~30 (20 headless) ms/frame

### List of Changes (release alpha v0.3.4)
21st June 2020
1. Fixed the science birds game (SB) and game playing interface (GPI) crash bug when the batch ground truth request sent but the shot is not being executed.
    - one ground truth will be returned in such case
2. Fixed SB crash bug when sending zoom out request sometimes
3. The game level will be fully zoomed out by default after loading
4. Changed configMeta.xml format  (see README.md for details)
5. Changed game level name format
6. Allow multiple agents to connect to one GPI.
    - SB do not need to be started by the user, instead, one SB instance will be started automatically by the GPI when an new agent is connected to the GPI 
    - A few command line arguments were added to the GPI (see README.md for details)
7. Removed unstable game levels from the sample game levels

### List of Changes (release alpha v0.3.3)
19th May 2020
1. Add a new requet to send a batch of ground truths per n frames after a shot

### List of Changes (release alpha v0.3.0)
7th Apr 2020
1. Add test harness which generates test trials and send specific requests to the agent to perform the test trial
2. Add 6 new states and 3 requests related to test harness 

### List of Changes (release alpha v0.2.1)
5th Mar 2020

1. Modified the method of loading novelty levels.
    - now novelty levels with different novlety levels/types can be loaded at the same time
    - the combination of the level sets can be flexibly rearranged  

### List of Changes (release alpha v0.2.0)
28th Feb 2020

1. Added capability of reading novlety level 1-3 with 1200 sample levels
    -  Level 1: new objects with 5 novelty type samples provided (100 levels for each)
    -  Level 2: change of parameters of objects with 5 novelty type samples provided (100 levels for each)
    -  Level 3: change of representation with 2 novelty type samples provided (100 levels for each)
    - The original non-onvelty levels are also provided for comparasion
    - Note: the source code of the novelty generator is not included in the release
    - The instruction of loading novelty levels is in the Novel Levels Loading section of README.md 
2. Fixed cshoot return shoot successfully indicator before the level is stable problem. 
    - now the return value for cshoot/pshoot will be returned once the not objects in level is moving
    -  now the return value for cfastshoot/pfastshoot will be returned after the shoot procedure is finished, i.e., the drag and tap operations are executed 
3. Fixed science birds error message display bug 

### List of Changes (release alpha v0.1.2)
21th Feb 2020

1. The agent now can register an observer agent on port 2006 which allows the user to request the screenshots/groundtruth from another thread.
    - the observer agent can only execute 6 commands: configure (1), DoScreenshot (11) and the four groundtruth related (61-64)
    - the demo code of using this function is in src/demo/naive_agent_groundtruth.py line 53-81 and 153-154


### List of Changes (release alpha v0.1.1) 
19th Feb 2020 

1. Protocol #23 (Get my score) format is changed
    - a 4 bytes array indicating the number of levels is added in front of the score bytes array

2. Naive agent and DQ agent are adapted

### List of Changes (release alpha v0.1) 

10th Feb 2020 

This is a brief introduction of what has been changed in this version. Please refer to the [README](https://gitlab.com/sail-on-anu/sciencebirdsframework_release/-/blob/release/alpha-0.1/README.md) file for details.

1. Speed up
    - a new protocol code is added to change the simulation speed of Unity
    - a speed of d $\in$ (0 - 50] is allowed
            - where d $\in$ (0, 1) means to slow down the simulator for 1/d times
            - d = 1 means the normal speed
            - d $\in$ (1, 50] means to speed up the simulation for d times  
    
2. The change of (noisy) groundtruth representation  
    - add trajectory points
    - change object type representation
        - object other than ground, slingshot
    - add colour distribution
3. Headless run
    - graphic-free science birds can be produced by a server build from Unity 
    - the headless run should not need any spectial command using the server build version of science birds
4. Baseline agents are added including:
    - Eagle Wings (Planning)
    - DQ agent (Deep Q learning)

5. Score changing problem after WON/LOST banner shown up is solved
6. Protocol code 13 (get best score) has been removed as it performs the same as 23 (get my score) given only one agent will play the game.
