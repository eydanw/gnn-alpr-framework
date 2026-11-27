resources = [
    {
        "id": 1,
        "type": "article",
        "estimated_time": 30,  # in minutes
    },
    {
        "id": 2,
        "type": "video",
        "estimated_time": 15,  # in minutes
    },
    {
        "id": 3,
        "type": "article",
        "estimated_time": 45,  # in minutes
    },
    {
        "id": 4,
        "type": "video",
        "estimated_time": 10,  # in minutes
    },
    {
        "id": 5,
        "type": "article",
        "estimated_time": 60,  # in minutes
    },
    {
        "id": 6,
        "type": "video",
        "estimated_time": 20,  # in minutes
    },
    {
        "id": 7,
        "type": "article",
        "estimated_time": 25,  # in minutes
    },
    {
        "id": 8,
        "type": "video",
        "estimated_time": 30,  # in minutes
    },
    {
        "id": 9,
        "type": "article",
        "estimated_time": 40,  # in minutes
    },
    {
        "id": 10,
        "type": "video",
        "estimated_time": 5,  # in minutes
    }
]


def calculate_min_max_resource_duration(resources) -> list:
    VIDEO_MIN_DURATION_PERCENTAGE = 0.4
    VIDEO_MAX_DURATION_PERCENTAGE = 2
    ARTICLE_MIN_DURATION_PERCENTAGE = 0.2
    ARTICLE_MAX_DURATION_PERCENTAGE = 3

    for resource in resources:
        if resource["type"] == "video" and "estimated_time" in resource:
            resource["min_duration"] = resource["estimated_time"] * VIDEO_MIN_DURATION_PERCENTAGE
            resource["max_duration"] = resource["estimated_time"] * VIDEO_MAX_DURATION_PERCENTAGE
        elif resource["type"] == "article" and "estimated_time" in resource:
            resource["min_duration"] = resource["estimated_time"] * ARTICLE_MIN_DURATION_PERCENTAGE
            resource["max_duration"] = resource["estimated_time"] * ARTICLE_MAX_DURATION_PERCENTAGE

    return resources


resources = calculate_min_max_resource_duration(resources)

mock_learners = [
    {
        "id": 1,
        "participation_counts": {"T": 5, "H": 10, "D": 3},
        "activity_counts": {"S": 4, "C": 2, "L": 8, "R": 1},
    },
    {
        "id": 2,
        "participation_counts": {"T": 12, "H": 8, "D": 5},
        "activity_counts": {"S": 10, "C": 6, "L": 12, "R": 3},
    },
    {
        "id": 3,
        "participation_counts": {"T": 20, "H": 15, "D": 10},
        "activity_counts": {"S": 30, "C": 12, "L": 40, "R": 8},
    },
    {
        "id": 4,
        "participation_counts": {"T": 1, "H": 0, "D": 0},
        "activity_counts": {"S": 0, "C": 0, "L": 1, "R": 0},
    },
    {
        "id": 5,
        "participation_counts": {"T": 50, "H": 40, "D": 20},
        "activity_counts": {"S": 100, "C": 80, "L": 60, "R": 50},
    },
    {
        "id": 6,
        "participation_counts": {"T": 8, "H": 6, "D": 2},
        "activity_counts": {"S": 12, "C": 4, "L": 6, "R": 1},
    },
    {
        "id": 7,
        "participation_counts": {"T": 4, "H": 2, "D": 1},
        "activity_counts": {"S": 5, "C": 2, "L": 3, "R": 0},
    },
    {
        "id": 8,
        "participation_counts": {"T": 15, "H": 12, "D": 8},
        "activity_counts": {"S": 20, "C": 10, "L": 15, "R": 5},
    },
    {
        "id": 9,
        "participation_counts": {"T": 7, "H": 9, "D": 4},
        "activity_counts": {"S": 14, "C": 7, "L": 12, "R": 2},
    },
    {
        "id": 10,
        "participation_counts": {"T": 3, "H": 2, "D": 1},
        "activity_counts": {"S": 3, "C": 1, "L": 2, "R": 0},
    },
]

learn_logs = [
    {
        "learner_id": 1,
        "resource_id": 2,
        "time_spent": 15,  # in minutes
    },
    {
        "learner_id": 1,
        "resource_id": 3,
        "time_spent": 20,  # in minutes
    },
    {
        "learner_id": 2,
        "resource_id": 1,
        "time_spent": 25,  # in minutes
    },
    {
        "learner_id": 2,
        "resource_id": 4,
        "time_spent": 10,  # in minutes
    },
    {
        "learner_id": 3,
        "resource_id": 5,
        "time_spent": 60,  # in minutes
    },
    {
        "learner_id": 3,
        "resource_id": 6,
        "time_spent": 15,  # in minutes
    },
    {
        "learner_id": 4,
        "resource_id": 7,
        "time_spent": 20,  # in minutes
    },
    {
        "learner_id": 5,
        "resource_id": 8,
        "time_spent": 30,  # in minutes
    },
    {
        "learner_id": 5,
        "resource_id": 9,
        "time_spent": 35,  # in minutes
    },
    {
        "learner_id": 6,
        "resource_id": 10,
        "time_spent": 5,  # in minutes
    },
    {
        "learner_id": 7,
        "resource_id": 1,
        "time_spent": 30,  # in minutes
    },
    {
        "learner_id": 8,
        "resource_id": 2,
        "time_spent": 15,  # in minutes
    },
    {
        "learner_id": 9,
        "resource_id": 3,
        "time_spent": 40,  # in minutes
    },
    {
        "learner_id": 10,
        "resource_id": 4,
        "time_spent": 10,  # in minutes
    },
]
