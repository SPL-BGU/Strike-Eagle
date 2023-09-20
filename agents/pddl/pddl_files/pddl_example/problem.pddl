(define (problem sample_problem)
    (:domain angry_birds_scaled)
    (:objects
        bird1 - bird
        pig1 - pig
    )
    (:init
        (= (angle) 2)
        (= (angle_rate) 0.5)


        (= (y_bird bird1) 200)
        (= (x_bird bird1) 10)
        (= (v_bird bird1) 100)
        (= (bird_id bird1) 0)
        (= (m_bird bird1) 1)

        (= (bird_type bird1) 0)
        (= (bird_radius bird1) 100)


        (= (y_pig pig1) 400)
        (= (x_pig pig1) 400)
        (= (pig_radius pig1) 100)
        (= (pig_life pig1) 1)
        (= (m_pig pig1) 0)

        (= (active_bird) 0)

        (= (bounce_count) 0)
        (= (gravity) 10)
    )
    (:goal
        ; Define your goal conditions here
        (and
            (pig_dead pig1)
        )
    )
)
