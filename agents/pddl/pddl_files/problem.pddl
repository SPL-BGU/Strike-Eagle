(define (problem sample_problem)
    (:domain angry_birds_scaled)
    (:objects
        bird_0 - bird
pig_0 - pig
    )
    (:init
        (= (angle) 90)
(= (angle_rad) 1.5707963267948966)
(= (angle_rate) 0.2)
(= (angle_bias) 0)
(= (cosine) 0 )
(= (sinus) 1 )
(= (bounce_count) 0)
(= (gravity) 85.87057732997192)
(= (active_bird) 0)
(= (ground_y_damper) 0.1)(= (ground_x_damper) 0.5)
(= (x_bird bird_0) 97)
(= (y_bird bird_0) 388)
(= (bird_id bird_0) 0)
(= (bird_type bird_0) 0)
(= (m_bird bird_0) 56)
(= (bird_radius bird_0) 3.5)
(= (v_bird bird_0) 180.51183368411057)
(= (bounce_count bird_0) 0)
(= (x_pig pig_0) 126.5)
(= (y_pig pig_0) 355)
(= (m_pig pig_0) 56)
(= (pig_radius pig_0) 3.5)
(= (pig_life pig_0) 1)
    )
    (:goal
        ; Define your goal conditions here
        (and
            (pig_dead pig_0)
        )
    )
)
