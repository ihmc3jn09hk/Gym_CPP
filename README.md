# LP_Plugin_SoftActorCritic

The Gym implemented CartPole for 1D and 2D continuous.

For the example shown below, it uses vision-based version of 2D continuous CartPole.

1D action: [ -1 < x < 1 ]

1D state : [ position, velocity, angle, angular velocity ]

2D action: [ -1 < x < 1, -1 < y < 1 ]

2D state : [ position_x, velocity_x, angle_x, angular velocity_x, position_y, velocity_y, angle_y, angular velocity_y ]

The CartPole VisionContinuous does not limit the cart position and linear velocity. So the cart is moving in an infinite plane or sphere as shown by the example.
The vision-based version cannot be used directly. A callback renderer function is needed which maybe provided in the future.

Example


![Demo](global.gif)

The states input show from the top-view (128x128). 

The color (Depth as alpha not shown here) input only.

The following scaled to 512x512 for illustration.


![State/Features seen by the AI](features_in.gif)
