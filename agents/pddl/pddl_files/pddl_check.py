from pddlpy import DomainProblem

# Load domain and problem files
domain_file = "domain.pddl"
problem_file = "problem.pddl"

domain = DomainProblem(domain_file, problem_file)

# # Initialize planner (e.g., FastDownward)
# planner = Planner(domain, "fast-downward")
#
# # Run the planner and get the plan
# plan, _ = planner.solve()
#
# if plan:
#     print("Plan found:")
#     for action in plan:
#         print(action)
# else:
#     print("No plan found.")
