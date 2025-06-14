(define (problem sample_problem)
    (:domain angry_birds_scaled)
    (:objects
        bird_0 - bird
pig_0 - pig
platform_0 - platform
    )
    (:init
        (= (angle) 90)
(= (angle_rad) 1.5707963267948966)
(= (angle_rate) 0.2)
(= (cosine) 0 )
(= (sinus) 1 )
(= (bounce_count) 0)
(= (gravity) 90)
(= (active_bird) 0)
(= (ground_y_damper) 0.1)(= (ground_x_damper) 0.5)
(= (x_bird bird_0) 97)
(= (y_bird bird_0) 388)
(= (bird_id bird_0) 0)
(= (bird_type bird_0) 0)
(= (m_bird bird_0) 56)
(= (bird_radius bird_0) 3.5)
(= (v_bird bird_0) 200)
(= (bounce_count bird_0) 0)
(= (x_pig pig_0) 360.5)
(= (y_pig pig_0) 354)
(= (m_pig pig_0) 56)
(= (pig_radius pig_0) 3.5)
(= (pig_life pig_0) 1)
(= (x_platform platform_0) 361.0)
(= (y_platform platform_0) 378.0)
(= (platform_width platform_0) 14)
(= (platform_height platform_0) 6)
    )
    (:goal
        ; Define your goal conditions here
        (and
            (pig_dead pig_0)
        )
    )
)
