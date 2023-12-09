(define
    (problem AngryBirds-level)
    (:domain AngryBirds-domain)
    (:objects
        bird1 bird2 - BIRD
        pig1 pig2 - PIG
        slingshot - SLINGSHOT
    )

    (:init
        (at-location bird1 slingshot)
        (in-sight pig1)
        (in-sight pig2)
        (exist bird1)
        (exist bird2)
        (exist pig1)
        (exist pig2)
        (next-in-line bird1 bird2)
    )

    (:goal
        (and
            (killed pig1)
            (killed pig2)
        )
    )

)