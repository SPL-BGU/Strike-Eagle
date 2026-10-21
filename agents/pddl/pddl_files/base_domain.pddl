(define (domain angry_birds_scaled)
    (:requirements :typing :disjunctive-preconditions :fluents :time :negative-preconditions :conditional-effects)
    (:types bird pig block platform external_agent)
    (:predicates (bird_released ?b - bird) (pig_dead ?p - pig) (angle_adjusted) (block_explosive ?bl - block) (pig_killed) (agent_dead ?ea - external_agent) (bird_tapped ?b - bird) (force_locked)
                (platform_slide_active ?b - bird) (bird_sliding ?b - bird ?pl - platform)
                (block_slide_active ?bl - block) (block_sliding ?bl - block ?pl - platform)
                (platform_chain ?from - platform ?to - platform) (platform_has_next ?pl - platform))

    (:functions (x_bird ?b - bird) (y_bird ?b - bird) (v_bird ?b - bird) (vx_bird ?b - bird) (vy_bird ?b - bird)  (m_bird ?b - bird) (bird_id ?b - bird) (bounce_count ?b - bird)
                (bird_type ?b - bird) ;; BIRD TYPES: RED=0, YELLOW=1, BLACK2, WHITE=3, BLUE=4 ;;
                (bird_radius ?b - bird)
                (bird_block_damage ?b - bird ?bl - block)
                (mod) (cosine) (sinus)
                (gravity) (angle_rate) (angle) (angle_rad) (angle_bias) (active_bird) (ground_y_damper)(ground_x_damper) (max_angle) (gravity_factor)  (min_angle)
                (base_life_wood_multiplier) (base_life_ice_multiplier) (base_life_stone_multiplier) (base_life_tnt_multiplier)
                (base_mass_wood_multiplier) (base_mass_ice_multiplier) (base_mass_stone_multiplier) (base_mass_tnt_multiplier)
                (meta_wood_multiplier) (meta_stone_multiplier) (meta_ice_multiplier) (meta_platform_size) ; TODO add: meta_block_size tnt_explosion_size bird_explosion_size
                (v_bird_multiplier) (min_force) (max_force) (force_rate)
                (x_pig ?p - pig) (y_pig ?p - pig) (pig_radius ?p - pig) (m_pig ?p - pig)
                (pig_life ?p - pig) (fall_damage) (explosion_damage)
                (x_platform ?pl - platform) (y_platform ?pl - platform) (platform_width ?pl - platform) (platform_height ?pl - platform)
                (platform_top_y_left ?pl - platform) (platform_top_y_right ?pl - platform) (platform_top_y_min ?pl - platform) (platform_top_y_max ?pl - platform)
                (platform_x_left ?pl - platform) (platform_slope ?pl - platform) (platform_slide_accel ?pl - platform) (platform_line_c ?pl - platform)
                (x_block ?bl - block) (y_block ?bl - block) (block_width ?bl - block) (block_height ?bl - block) (block_life ?bl - block) (block_mass ?bl - block) (block_stability ?bl - block) (v_block ?bl - block) (vx_block ?bl - block) (vy_block ?bl - block)
                (points_score)
                (x_agent ?ea - external_agent) (y_agent ?ea - external_agent) (agent_height ?ea - external_agent) (agent_width ?ea - external_agent)
                (vx_agent ?ea - external_agent) (vy_agent ?ea - external_agent) (timing_agent ?ea - external_agent)
                (x_max_border ?ea - external_agent) (x_min_border ?ea - external_agent) (y_max_border ?ea - external_agent) (y_min_border ?ea - external_agent)
                ; (x_exclusion ?ea - external_agent) (y_exclusion ?ea - external_agent) (exclusion_height ?ea - external_agent) (exclusion_width ?ea - external_agent)
                (agent_type ?ea - external_agent) ;; AGENT TYPES: MAGICIAN=0, WIZARD=1, BUTTERFLY=2, WORM=3, unknown=4,
                ;;
                ; (x_marker ?mp - marker_point) (y_marker ?mp - marker_point) ;; POINTS WHICH THE AGENTS MOVE AROUND (made explicit to avoid chacking all blocks and platforms, CHECK WITH OTHERS?).
                ;;
                ;; WOOD LIFE = 0.75   WOOD MASS COEFFICIENT = 0.375 ;; ICE LIFE = 0.75   ICE MASS COEFFICIENT = 0.375 ;; STONE LIFE = 1.2   STONE MASS COEFFICIENT = 0.375
                ;; WOOD LIFE MULTIPLIER = 1.0 ;; ICE LIFE MULTIPLIER = 0.5 ;; STONE LIFE MULTIPLIER = 2.0
                ;; THESE VALUES NEED TO BE VERIFIED
                ;; SEEMS LIKE THE SIZE AND SHAPE OF EACH BLOCK HAVE DIFFERENT LIFE VALUES WHICH ARE THEN MULTIPLIED BY THE MATERIAL LIFE MULTIPLIER
                ;; FOR NOW I WILL ASSIGN BLOCK_LIFE == 1.0 * MATERIAL_MULTIPLIER WHICH WILL BE ASSIGNED AUTOMATICALLY IN THE PROBLEM FILE VIA HYDRA.
    )

    (:process increasing_angle
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_released ?b))
            (not (angle_adjusted))
            (<= (angle) (max_angle))
            (>= (angle) (min_angle))
        )
        :effect (and
            (decrease (angle) (* #t (angle_rate)))
            ; Use the post-decrement angle: effects read pre-step values, so (angle) here
            ; is still the old angle and the trig would lag the parsed dial by #t*rate.
            (assign (angle_rad) (* (- (- (angle) (* #t (angle_rate))) (angle_bias))  0.01745329252))
            ;(assign (cosine) (- (+ (- 1 (/ (* (angle_rad) (angle_rad)) 2)) (/ (* (* (* (angle_rad) (angle_rad)) (angle_rad)) (angle_rad)) 24))))
            (
	assign
	(cosine)
    (
		-
		(
			+
			(
				-
				1
				(/ (* (angle_rad) (angle_rad)) 2)
			)
			(/ (* (* (* (angle_rad) (angle_rad)) (angle_rad)) (angle_rad)) 24)
		)
		(
			/
			(
				*
				(
					*
					(
						*
						(
							*
							(
							    *
							    (angle_rad)
							    (angle_rad)
							)
							(angle_rad)
						)
						(angle_rad)
					)
					(angle_rad)
				)
				(angle_rad)
			)
			720
		)
	)
)

            (assign (sinus) (+ (- (angle_rad) (/ (* (* (angle_rad) (angle_rad) ) (angle_rad) ) 6 ) ) (/ (* (* (* (* (angle_rad) (angle_rad) ) (angle_rad) ) (angle_rad) ) (angle_rad) ) 120 ) ) )

        )
    )

    (:process decreasing_force
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_released ?b))
            (not (force_locked))
            (<= (v_bird_multiplier) (max_force))
            (>= (v_bird_multiplier) (min_force))
        )
        :effect (and
            (decrease (v_bird_multiplier) (* #t (force_rate)))
        )
    )

    (:action set_force
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_released ?b))
            (not (force_locked))
        )
        :effect (force_locked)
    )

    (:process flying
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (not (platform_slide_active ?b))
            (> (y_bird ?b) 0)
        )
        :effect (and
            (increase (y_bird ?b) (* (/ #t 100) (* 1.0 (vy_bird ?b))))
            (decrease (vy_bird ?b) (* (/ #t 100) (* 1.0 (gravity)) ))
            (increase (x_bird ?b) (* (/ #t 100) (* 1.0 (vx_bird ?b))))
            (assign (mod) 0)
        )
    )

    (:process platform_sliding
        :parameters (?b - bird ?pl - platform)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (platform_slide_active ?b)
            (bird_sliding ?b ?pl)
            (> (vx_bird ?b) 0)
            (<= (- (x_bird ?b) (* (bird_radius ?b) 0.35)) (+ (x_platform ?pl) (/ (platform_width ?pl) 2)))
            (>= (+ (x_bird ?b) (* (bird_radius ?b) 0.35)) (- (x_platform ?pl) (/ (platform_width ?pl) 2)))
        )
        :effect (and
            (assign (y_bird ?b) (+ (+ (platform_line_c ?pl) (* (platform_slope ?pl) (x_bird ?b))) (* (bird_radius ?b) 0.35)))
            (assign (vy_bird ?b) 0)
            (increase (x_bird ?b) (* (/ #t 100) (* 1.0 (vx_bird ?b))))
            (increase (vx_bird ?b) (* (/ #t 100) (* (gravity) (platform_slide_accel ?pl))))
            (decrease (vx_bird ?b) (* (/ #t 100) {SE-platform-friction}))
            (assign (mod) 0)
        )
    )

    (:action pa-twang
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (not (bird_released ?b))
            (not (angle_adjusted))
            (force_locked)
        )
        :effect (and
            (assign (vx_bird ?b) (* (* (v_bird ?b) (v_bird_multiplier)) (cosine) ) )
            (assign (vy_bird ?b) (* (* (v_bird ?b) (v_bird_multiplier)) (sinus) ) )
            (decrease (x_bird ?b) (* 16 (cosine))) ;; define this values
            (decrease (y_bird ?b) (* 12 (sinus))) ;; define this values

            (bird_released ?b)
            (angle_adjusted)
            ;(decrease (points_score) 10000)
        )
    )

     (:event collision_ground
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (<= (y_bird ?b) 0)
            (> (vx_bird ?b) 0)
            ;(> (v_bird ?b) 20)

        )
        :effect (and
            {SE-collision-ground-effect}
        )
    )

    (:event explode_bird
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (or (>= (bounce_count ?b) 3)
                (>= (x_bird ?b) 800)
            )
            (angle_adjusted)
            ; (<= (v_bird) 20)
        )
        :effect (and
            (assign (angle) (min_angle))
            (assign (active_bird) (+ (active_bird) 1) )
            (not (angle_adjusted))
        )
    )

    (:event collision_pig_kill
        :parameters (?b - bird ?p - pig)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (v_bird ?b) 0)
            (>=
                (*
                    (-
                        (+
                            (bird_radius ?b)
                            (pig_radius ?p)
                        )
                        4.0
                    )
                    (-
                        (+
                            (bird_radius ?b)
                            (pig_radius ?p)
                        )
                        4.0
                    )
                )
                (+
                    (*
                        (-
                            (x_bird ?b)
                            (x_pig ?p)
                        )
                        (-
                           (x_bird ?b)
                           (x_pig ?p)
                        )
                    )
                    (*
                        (-
                            (y_bird ?b)
                            (y_pig ?p)
                        )
                        (-
                            (y_bird ?b)
                            (y_pig ?p)
                        )
                    )
                )
            )
            (not (pig_dead ?p))
            (<= (pig_life ?p) (v_bird ?b))
        )
        :effect (and
            (assign (vx_bird ?b) (- (vx_bird ?b) (* (/ (* 2 (m_pig ?p)) (+ (m_bird ?b) (m_pig ?p))) (*
                (/
                    (+ (+ (- (- (- (- (+ (* (vx_bird ?b) (x_bird ?b)) (* (vy_bird ?b) (y_bird ?b))) (* (vx_bird ?b) (x_pig ?p))) (* (vy_bird ?b) (y_pig ?p))) (* 0 (x_bird ?b))) (* 0 (y_bird ?b))) (* 0 (x_pig ?p))) (* 0 (y_pig ?p)))
                    (+ (+ (- (- (- (- (+ (* (x_pig ?p) (x_pig ?p)) (* (y_pig ?p) (y_pig ?p))) (* (x_pig ?p) (x_bird ?b))) (* (y_pig ?p) (y_bird ?b))) (* (x_bird ?b) (x_pig ?p))) (* (y_bird ?b) (y_pig ?p))) (* (x_bird ?b) (x_bird ?b))) (* (y_bird ?b) (y_bird ?b)))
                )
                (- (x_pig ?p) (x_bird ?b)) )
            )  ) )

            (assign (vy_bird ?b) (- (vy_bird ?b) (* (/ (* 2 (m_pig ?p)) (+ (m_bird ?b) (m_pig ?p))) (*
                (/
                    (+ (+ (- (- (- (- (+ (* (vx_bird ?b) (x_bird ?b)) (* (vy_bird ?b) (y_bird ?b))) (* (vx_bird ?b) (x_pig ?p))) (* (vy_bird ?b) (y_pig ?p))) (* 0 (x_bird ?b))) (* 0 (y_bird ?b))) (* 0 (x_pig ?p))) (* 0 (y_pig ?p)))
                    (+ (+ (- (- (- (- (+ (* (x_pig ?p) (x_pig ?p)) (* (y_pig ?p) (y_pig ?p))) (* (x_pig ?p) (x_bird ?b))) (* (y_pig ?p) (y_bird ?b))) (* (x_bird ?b) (x_pig ?p))) (* (y_bird ?b) (y_pig ?p))) (* (x_bird ?b) (x_bird ?b))) (* (y_bird ?b) (y_bird ?b)))
                )
                (- (y_pig ?p) (y_bird ?b)) )
            )  ) )
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )
    (:event collision_block_bounce_off
        :parameters (?b - bird ?bl - block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (v_bird ?b) 0)
            (>= (+ (x_bird ?b) (bird_radius ?b)) (- (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (<= (- (x_bird ?b) (bird_radius ?b)) (+ (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (>= (+ (y_bird ?b) (bird_radius ?b)) (- (y_block ?bl) (/ (block_height ?bl) 2) ) )
            (<= (- (y_bird ?b) (bird_radius ?b)) (+ (y_block ?bl) (/ (block_height ?bl) 2) ) )
            (> (block_life ?bl) 0)
            ; Bird has enough velocity to push the block (velocity >= stability threshold)
            (>= (v_bird ?b) (block_stability ?bl))
            ; Bird can't break through (life check fails) - bounces instead
            (> (* (block_life ?bl) (block_life ?bl) ) ( * ( + ( * (vx_bird ?b) (vx_bird ?b) ) (	* (vy_bird ?b) (vy_bird ?b) ) ) ( * (bird_block_damage ?b ?bl) (bird_block_damage ?b ?bl) ) ) )
        )
        :effect (and
            ; Bird bounces off; block gets the learned push ratio of the bird's velocity
            (assign (vx_block ?bl) (* {SE-block-push-ratio} (vx_bird ?b)))
            (assign (vy_block ?bl) (* {SE-block-push-ratio} (vy_bird ?b)))
            (assign (v_block ?bl) (* {SE-block-push-ratio} (v_bird ?b)))
            ; Bird stops after bounce
            (assign (v_bird ?b) 0)
            (assign (vx_bird ?b) 0)
            (assign (vy_bird ?b) 0)
            (assign (bounce_count ?b) 3)
        )
    )


    (:event collision_block
        :parameters (?b - bird ?bl - block)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (v_bird ?b) 0)
            (>= (+ (x_bird ?b) (bird_radius ?b)) (- (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (<= (- (x_bird ?b) (bird_radius ?b)) (+ (x_block ?bl) (/ (block_width ?bl) 2) ) )
            (>= (+ (y_bird ?b) (bird_radius ?b)) (- (y_block ?bl) (/ (block_height ?bl) 2) ) )
            (<= (- (y_bird ?b) (bird_radius ?b)) (+ (y_block ?bl) (/ (block_height ?bl) 2) ) )
            (> (block_life ?bl) 0)
            (<= (* (block_life ?bl) (block_life ?bl) ) ( * ( + ( * (vx_bird ?b) (vx_bird ?b) ) (	* (vy_bird ?b) (vy_bird ?b) ) ) ( * (bird_block_damage ?b ?bl) (bird_block_damage ?b ?bl) ) ) ) )
        :effect (and
            ; Bird breaks through block; block gets the learned push ratio of the bird's velocity
            (assign (vx_block ?bl) (* {SE-block-push-ratio} (vx_bird ?b)))
            (assign (vy_block ?bl) (* {SE-block-push-ratio} (vy_bird ?b)))
            (assign (v_block ?bl) (* {SE-block-push-ratio} (v_bird ?b)))
            ; Bird stops after collision
            (assign (bounce_count ?b) 3)
        )
    )

    (:event remove_unsupported_block
        :parameters (?bl_bottom - block ?bl_top - block)
        :precondition (and
            (> (block_life ?bl_top) 0)
            (> (block_stability ?bl_top) 0)
            (or
                (<= (block_life ?bl_bottom) 0)
                (<= (block_stability ?bl_bottom) 0)
            )
            (<= (x_block ?bl_bottom) (+ (x_block ?bl_top) (/ (block_width ?bl_top) 2) ) )
            (>= (x_block ?bl_bottom) (- (x_block ?bl_top) (/ (block_width ?bl_top) 2) ) )
            (<= (y_block ?bl_bottom) (- (y_block ?bl_top) (/ (block_height ?bl_top) 2) ) )
        )
        :effect (and
            (assign (block_life ?bl_top) (- (block_life ?bl_top) 100) )
            (assign (y_block ?bl_top) (/ (block_height ?bl_top) 2) )
            (assign (block_stability ?bl_top) 0)
            (increase (points_score) 500)
        )
    )

    (:event explode_block
        ; The actual explosion radius is 50 (in whatever units this is??). We use a value of 100 to help model chain reactions.
        :parameters (?bl_tnt - block ?bl_near - block)
        :precondition (and
            (> (block_life ?bl_near) 0)
            (> (block_stability ?bl_near) 0)
            (<= (- (x_block ?bl_tnt) (x_block ?bl_near)) 100 )
            (>= (- (x_block ?bl_tnt) (x_block ?bl_near)) -100 )
            (<= (- (y_block ?bl_tnt) (y_block ?bl_near)) 100 )
            (>= (- (y_block ?bl_tnt) (y_block ?bl_near)) -100 )
            (block_explosive ?bl_tnt)
            ; (<= (block_stability ?bl_tnt) 0)
            (<= (block_life ?bl_tnt) 0)

        )
        :effect (and
            (assign (block_life ?bl_near) 0)
            (assign (block_stability ?bl_near) 0)
            (increase (points_score) 1000)
        )
    )

    (:event explode_pig
        :parameters (?bl_tnt - block ?p - pig)
        :precondition (and
            (not (pig_dead ?p))
            (block_explosive ?bl_tnt)
            ; (<= (block_stability ?bl_tnt) 0)
            (<= (block_life ?bl_tnt) 0)
            (<= (- (x_block ?bl_tnt) (x_pig ?p)) 50 )
            (>= (- (x_block ?bl_tnt) (x_pig ?p)) -50 )
            (<= (- (y_block ?bl_tnt) (y_pig ?p)) 50 )
            (>= (- (y_block ?bl_tnt) (y_pig ?p)) -50 )
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    (:event remove_unsupported_pig
        :parameters (?bl_bottom - block ?p - pig)
        :precondition (and
        	(not (pig_dead ?p))
            (or
                (< (block_life ?bl_bottom) 0)
                (<= (block_stability ?bl_bottom) 0)
            )
            (<= (x_pig ?p) (+ (x_block ?bl_bottom) (/ (block_width ?bl_bottom) 2) ) )
            (>= (x_pig ?p) (- (x_block ?bl_bottom) (/ (block_width ?bl_bottom) 2) ) )
            (>= (y_pig ?p) (+ (y_block ?bl_bottom) (/ (block_height ?bl_bottom) 2) ) )
            (<= (y_pig ?p) (+ (y_block ?bl_bottom) (+ (block_height ?bl_bottom) (pig_radius ?p))) )
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    ; Rolling block kills pig - RE-ENABLED for pig_dead goal with pig in problem
    (:event collision_block_pig_kill
        :parameters (?bl - block ?p - pig)
        :precondition (and
            (not (pig_dead ?p))
            (> (block_life ?bl) 0)
            (> (v_block ?bl) 10)
            (>= (+ (x_block ?bl) (/ (block_width ?bl) 2)) (- (x_pig ?p) (pig_radius ?p)))
            (<= (- (x_block ?bl) (/ (block_width ?bl) 2)) (+ (x_pig ?p) (pig_radius ?p)))
            (>= (+ (y_block ?bl) (/ (block_height ?bl) 2)) (- (y_pig ?p) (pig_radius ?p)))
            (<= (- (y_block ?bl) (/ (block_height ?bl) 2)) (+ (y_pig ?p) (pig_radius ?p)))
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    (:event collision_platform
        :parameters (?b - bird ?pl - platform)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
            (not (platform_slide_active ?b))
            (> (v_bird ?b) 0)

            ; x-checks keep the 1.1× margin (grazing the side of a platform is
            ; legitimately hard in the game). y-checks use the true bird radius
            ; so overhung-pig layouts (pig sitting on a platform with a roof
            ; platform above) have a real navigable slot — the previous 1.1× y
            ; margin made template-4 single_force levels systematically report
            ; "Problem unsolvable" in ENHSP (see _unsolvable_snapshots).
            (<= (- (x_bird ?b)  (* (bird_radius ?b) 1.1)) (+ (x_platform ?pl) (/ (platform_width ?pl) 2 ) ) )
            (>= (+ (x_bird ?b)  (* (bird_radius ?b) 1.1)) (- (x_platform ?pl) (/ (platform_width ?pl) 2 ) ) )
            ; y-checks are against the top-edge line at the bird's x (equals the
            ; flat top for non-sloped platforms).
            (>= (+ (y_bird ?b)  (bird_radius ?b)) (- (+ (platform_line_c ?pl) (* (platform_slope ?pl) (x_bird ?b))) (platform_height ?pl)))
            (<= (- (y_bird ?b)  (bird_radius ?b)) (+ (platform_line_c ?pl) (* (platform_slope ?pl) (x_bird ?b))))
            ; First contact only — approach from above the stand line.
            (> (y_bird ?b) (+ (+ (platform_line_c ?pl) (* (platform_slope ?pl) (x_bird ?b))) (* (bird_radius ?b) 0.35)))
        )
        :effect (and
            {SE-collision-platform-effect}
        )
    )

    ; Slide continues onto a touching platform (precomputed platform_chain):
    ; no landing damping, no x jump, y snaps to the next platform's top line.
    (:event platform_slide_handoff
        :parameters (?b - bird ?from - platform ?to - platform)
        :precondition (and
            (platform_slide_active ?b)
            (bird_sliding ?b ?from)
            (platform_chain ?from ?to)
            (> (vx_bird ?b) 0)
            (> (- (x_bird ?b) (* (bird_radius ?b) 0.35)) (+ (x_platform ?from) (/ (platform_width ?from) 2)))
        )
        :effect (and
            (not (bird_sliding ?b ?from))
            (bird_sliding ?b ?to)
            (assign (y_bird ?b) (+ (+ (platform_line_c ?to) (* (platform_slope ?to) (x_bird ?b))) (* (bird_radius ?b) 0.35)))
            (assign (vy_bird ?b) 0)
        )
    )

    ; Bird stopped by friction: it rests on the platform and the shot ends
    ; (slide flags stay set so the flying process does not drop it through).
    (:event platform_slide_stop
        :parameters (?b - bird ?pl - platform)
        :precondition (and
            (platform_slide_active ?b)
            (bird_sliding ?b ?pl)
            (<= (vx_bird ?b) 0)
            (< (bounce_count ?b) 3)
        )
        :effect (and
            (assign (vx_bird ?b) 0)
            (assign (v_bird ?b) 0)
            (assign (bounce_count ?b) 3)
        )
    )

    (:event platform_slide_end
        :parameters (?b - bird ?pl - platform)
        :precondition (and
            (platform_slide_active ?b)
            (bird_sliding ?b ?pl)
            (or
                (and
                    (not (platform_has_next ?pl))
                    (> (- (x_bird ?b) (* (bird_radius ?b) 0.35)) (+ (x_platform ?pl) (/ (platform_width ?pl) 2)))
                )
                (< (+ (x_bird ?b) (* (bird_radius ?b) 0.35)) (- (x_platform ?pl) (/ (platform_width ?pl) 2)))
            )
        )
        :effect (and
            (not (platform_slide_active ?b))
            (not (bird_sliding ?b ?pl))
        )
    )

    ; Block collides with platform and starts rolling
    ; Y-check allows blocks resting ON TOP of platform (within half block height)
    (:event collision_block_platform
        :parameters (?bl - block ?pl - platform)
        :precondition (and
            (not (block_slide_active ?bl))
            (> (block_life ?bl) 0)
            (> (v_block ?bl) 0)
            ; X overlap check
            (<= (- (x_block ?bl) (/ (block_width ?bl) 2)) (+ (x_platform ?pl) (/ (platform_width ?pl) 2)))
            (>= (+ (x_block ?bl) (/ (block_width ?bl) 2)) (- (x_platform ?pl) (/ (platform_width ?pl) 2)))
            ; Y check: block bottom within half-block-height of platform top
            (<= (- (y_block ?bl) (/ (block_height ?bl) 2)) (+ (platform_top_y_max ?pl) (/ (block_height ?bl) 2)))
            (>= (- (y_block ?bl) (/ (block_height ?bl) 2)) (- (platform_top_y_min ?pl) (/ (platform_height ?pl) 2)))
        )
        :effect (and
            {SE-block-platform-roll-effect}
        )
    )

    ; Block continues rolling on platform
    (:process block_platform_rolling
        :parameters (?bl - block ?pl - platform)
        :precondition (and
            (block_slide_active ?bl)
            (block_sliding ?bl ?pl)
            (> (vx_block ?bl) 0)
            (> (v_block ?bl) 0)
            (<= (- (x_block ?bl) (/ (block_width ?bl) 2)) (+ (x_platform ?pl) (/ (platform_width ?pl) 2)))
            (>= (+ (x_block ?bl) (/ (block_width ?bl) 2)) (- (x_platform ?pl) (/ (platform_width ?pl) 2)))
        )
        :effect (and
            ; Keep block on the platform's top-edge line at its x
            (assign (y_block ?bl) (+ (+ (platform_line_c ?pl) (* (platform_slope ?pl) (x_block ?bl))) (/ (block_height ?bl) 2)))
            (assign (vy_block ?bl) 0)
            (increase (x_block ?bl) (* (/ #t 100) (* 1.0 (vx_block ?bl))))
            (increase (vx_block ?bl) (* (/ #t 100) (* (gravity) (platform_slide_accel ?pl))))
            (decrease (vx_block ?bl) (* (/ #t 100) {SE-platform-friction}))
            (assign (v_block ?bl) (vx_block ?bl))
            (assign (mod) 0)
        )
    )

    ; Roll continues onto a touching platform (precomputed platform_chain):
    ; no damping, y snaps to the next platform's top line.
    (:event block_platform_roll_handoff
        :parameters (?bl - block ?from - platform ?to - platform)
        :precondition (and
            (block_slide_active ?bl)
            (block_sliding ?bl ?from)
            (platform_chain ?from ?to)
            (> (vx_block ?bl) 0)
            (> (- (x_block ?bl) (/ (block_width ?bl) 2)) (+ (x_platform ?from) (/ (platform_width ?from) 2)))
        )
        :effect (and
            (not (block_sliding ?bl ?from))
            (block_sliding ?bl ?to)
            (assign (y_block ?bl) (+ (+ (platform_line_c ?to) (* (platform_slope ?to) (x_block ?bl))) (/ (block_height ?bl) 2)))
            (assign (vy_block ?bl) 0)
        )
    )

    ; Block stops rolling
    (:event block_platform_roll_end
        :parameters (?bl - block ?pl - platform)
        :precondition (and
            (block_slide_active ?bl)
            (block_sliding ?bl ?pl)
            (or
                (<= (vx_block ?bl) 0)
                (<= (v_block ?bl) 0)
                (and
                    (not (platform_has_next ?pl))
                    (> (- (x_block ?bl) (/ (block_width ?bl) 2)) (+ (x_platform ?pl) (/ (platform_width ?pl) 2)))
                )
                (< (+ (x_block ?bl) (/ (block_width ?bl) 2)) (- (x_platform ?pl) (/ (platform_width ?pl) 2)))
            )
        )
        :effect (and
            (not (block_slide_active ?bl))
            (not (block_sliding ?bl ?pl))
        )
    )

    ;; BIRD TYPES: RED=0, YELLOW=1, BLACK=2, WHITE=3, BLUE=4 ;;

    (:action yellow_bird_action
        :parameters (?b - bird)
        :precondition (and
        	(= (active_bird) (bird_id ?b))
      	    (bird_released ?b)
            (not (bird_tapped ?b))
            (= (bounce_count ?b) 0)
            (< (x_bird ?b) 800)
            (= (bird_type ?b) 1)
        )
        :effect (and
        	(assign (vx_bird ?b) (* (vx_bird ?b) 2))
        	(assign (vy_bird ?b) (* (vy_bird ?b) 2))
      	    (assign (v_bird ?b) (* (v_bird ?b) 2))
      	    (bird_tapped ?b)
  	    )
    )

    (:action white_bird_action
        :parameters (?b - bird)
        :precondition (and
      	    (= (active_bird) (bird_id ?b))
      	    (bird_released ?b)
      	    (not (bird_tapped ?b))
      	    (= (bounce_count ?b) 0)
            (< (x_bird ?b) 800)
            (= (bird_type ?b) 3)
        )
        :effect (and
      	    (assign (vx_bird ?b) 0)
      	    (assign (vy_bird ?b) 0)
      	    (assign (bird_radius ?b) (/ (bird_radius ?b) 1.5))
      	    (bird_tapped ?b)
            (assign (bird_radius ?b) (/ (bird_radius ?b) 2))
  	    )
    )

    (:action black_bird_action
        :parameters (?b - bird)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (bird_released ?b)
      	    (not (bird_tapped ?b))
      	    (< (bounce_count ?b) 3)
            (< (x_bird ?b) 800)
            (= (bird_type ?b) 2)
        )
        :effect (and
            (bird_tapped ?b)
            (assign (bounce_count ?b) 3)
        )
    )


    (:event explode_block_from_bird
        :parameters (?b - bird ?bl_near - block)
        :precondition (and
        	(= (active_bird) (bird_id ?b))
        	; (or
      		(= (bird_type ?b) 2)
      		(> (bounce_count ?b) 0)
      			; (and (= (bird_type ?b) 3) (= (bounce_count ?b) 1) (bird_tapped ?b) )
  			; )
            (> (block_stability ?bl_near) 0)
            (> (block_life ?bl_near) 0)
            (<= (- (x_bird ?b) (x_block ?bl_near)) 50 )
            (>= (- (x_bird ?b) (x_block ?bl_near)) -50 )
            (<= (- (y_bird ?b) (y_block ?bl_near)) 50 )
            (>= (- (y_bird ?b) (y_block ?bl_near)) -50 )
        )
        :effect (and
            (assign (block_life ?bl_near) 0)
            (assign (block_stability ?bl_near) 0)
        )
        )
    )

    (:event explode_pig_from_bird
        :parameters (?b - bird ?p - pig)
        :precondition (and
        	(= (active_bird) (bird_id ?b))
            (or
                (= (bird_type ?b) 2)
                (= (bounce_count ?b) 3)
                (platform_slide_active ?b)
            )
            (not (pig_dead ?p))
            (<= (- (x_bird ?b) (x_pig ?p)) 50 )
            (>= (- (x_bird ?b) (x_pig ?p)) -50 )
            (<= (- (y_bird ?b) (y_pig ?p)) 50 )
            (>= (- (y_bird ?b) (y_pig ?p)) -50 )
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )



    (:event explode_pig_from_egg
        :parameters (?b - bird ?p - pig)
        :precondition (and
        	(= (active_bird) (bird_id ?b))
        	; (or
      		(= (bird_type ?b) 3)
      		(= (bounce_count ?b) 3)
      			; (and (= (bird_type ?b) 3) (= (bounce_count ?b) 1) (bird_tapped ?b) )
  			; )
            (not (pig_dead ?p))
            (<= (- (x_bird ?b) (x_pig ?p)) 40 )
            (>= (- (x_bird ?b) (x_pig ?p)) -40 )
            (<= (- (y_bird ?b) (y_pig ?p)) 40 )
            (>= (- (y_bird ?b) (y_pig ?p)) -40 )
            (bird_tapped ?b)
        )
        :effect (and
            (pig_dead ?p)
            (pig_killed)
            (increase (points_score) 5000)
        )
    )

    ;; EXTERNAL AGENT EVENTS
    ;; AGENT TYPES: MAGICIAN=0, WIZARD=1, BUTTERFLY=2, WORM=3

    (:process agent_movement
        :parameters (?ea - external_agent)
        :precondition (and
            (angle_adjusted)
            (not (agent_dead ?ea))
        )
        :effect (and
            (increase (x_agent ?ea) (* #t (* 1.0 (vx_agent ?ea))))
            (increase (y_agent ?ea) (* #t (* 1.0 (vy_agent ?ea))))
            (increase (timing_agent ?ea) (* #t 200))
        )
    )

    (:event agent_1_timed_change_direction
        :parameters (?ea - external_agent)
        :precondition (and
            (angle_adjusted)
            (not (agent_dead ?ea))
            (= (agent_type ?ea) 1)
            (>= (timing_agent ?ea) 1000)
        )
        :effect (and
            (assign (vx_agent ?ea) (* (vx_agent ?ea) -1))
            (assign (vy_agent ?ea) (* (vy_agent ?ea) -1))
        )
    )

    (:event agent_x_border_change_direction
        :parameters (?ea - external_agent)
        :precondition (and
            (angle_adjusted)
            (not (agent_dead ?ea))
            (or
                (and (> (vx_agent ?ea) 0) (>= (+ (x_agent ?ea) (/ (agent_width ?ea) 2)) (x_max_border ?ea)) )
                (and (< (vx_agent ?ea) 0) (<= (- (x_agent ?ea) (/ (agent_width ?ea) 2)) (x_min_border ?ea)) )
            )
        )
        :effect (and
            (assign (vx_agent ?ea) (* (vx_agent ?ea) -1))
        )
    )

    (:event agent_y_border_change_direction
        :parameters (?ea - external_agent)
        :precondition (and
            (angle_adjusted)
            (not (agent_dead ?ea))
            (or
                (and (> (vy_agent ?ea) 0) (>= (+ (y_agent ?ea) (/ (agent_height ?ea) 2)) (y_max_border ?ea)) )
                (and (< (vy_agent ?ea) 0) (<= (- (y_agent ?ea) (/ (agent_height ?ea) 2)) (y_min_border ?ea)) )
            )
        )
        :effect (and
            (assign (vy_agent ?ea) (* (vy_agent ?ea) -1))
        )
    )

    (:event collision_agent
        :parameters (?b - bird ?ea - external_agent)
        :precondition (and
            (= (active_bird) (bird_id ?b))
            (> (v_bird ?b) 0)
            (<= (- (x_bird ?b) (bird_radius ?b)) (+ (x_agent ?ea) (/ (agent_width ?ea) 1.75) ) )
            (>= (+ (x_bird ?b) (bird_radius ?b)) (- (x_agent ?ea) (/ (agent_width ?ea) 1.75) ) )
            (>= (+ (y_bird ?b) (bird_radius ?b)) (- (y_agent ?ea) (/ (agent_height ?ea) 1.75) ) )
            (<= (- (y_bird ?b) (bird_radius ?b)) (+ (y_agent ?ea) (/ (agent_height ?ea) 1.75) ) )
        )
        :effect (and
            (assign (v_bird ?b) 0)
            (assign (vx_bird ?b) 0)
            (assign (vy_bird ?b) 0)
            (assign (bounce_count ?b) 3)
        )
    )
    (:action switch_bird
        :parameters (?b1 - bird)
        :precondition (and
            (bird_released ?b1)
      	    (= (active_bird) (bird_id ?b1))
      	    (= (vx_bird ?b1) 0)
        )
        :effect (and
            (increase (active_bird) 1)
            (not (angle_adjusted))
            (assign (angle) 0)

        )
    )


)
