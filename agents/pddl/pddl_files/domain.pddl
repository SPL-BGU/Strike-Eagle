(define (domain AngryBirds-domain)
    (:requirements :strips :typing)
    (:types
        STRUCTURE BIRD PIG SLINGSHOT - OBJECT
        ; WOOD STONE ICE - STRUCTURE
        ; DIRECTION
    )


    ; (:constants
    ;     up
    ;     down
    ;     left
    ;     right
    ; )




    (:predicates
        (in-motion ?o - OBJECT)
        ; (in-motion-dir ?o - OBJECT ?in - DIRECTION)
        (at-location ?o1 - OBJECT ?o2 - OBJECT)
        (exist ?o - OBJECT)
        (killed ?o - OBJECT)
        ; (in-sight ?object)                      ; Object is in sight of the slingshot
        (shot ?b - BIRD)                        ; bird is shot
        (next ?b1 - BIRD ?b2 - BIRD)            ; bird2 is next (after) bird1 in shooting order

    )



    (:action shot-bird
        :parameters (?bird - BIRD ?o - OBJECT  ?slingshot - SLINGSHOT)
        :precondition
        (
            and
            (at-location ?bird ?slingshot)
        )
        :effect
        (
            and
                (not (at-location ?bird ?slingshot))
                (at-location ?bird ?o)
                (in-motion ?bird)
                ; (in-motion-dir ?bird right)
                (shot ?bird)
        )
    )

    (:action hit
        :parameters(?object1 - OBJECT ?object2 - OBJECT )
        :precondition(
            and
            (at-location ?object1 ?object2)
            (in-motion ?object1)
            (exist ?object1)
            (exist ?object2)
        )
        :effect (
            and
              (killed ?object2)
        )

    )


    (:action fetch-bird
        :parameters(?bird1 - BIRD ?bird2 - BIRD ?slingshot - SLINGSHOT)
        :precondition(
            and
            (next ?bird1 ?bird2)
            (shot ?bird1)
        )
        :effect(
            and
            (at-location ?bird2 ?slingshot)
        )
    )

)
