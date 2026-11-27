from infrastructure.db.model.engine import get_session_ctx


async def load_learner_profile():
    async with get_session_ctx() as session:
        result = await session.execute("SELECT * FROM elearningapp.user_profile")
        learner_profiles = result.mappings().all()
        print(f"Load {len(learner_profiles)} learner profiles")
        return learner_profiles
async def load_lo():
    async with get_session_ctx() as session:
        result = await session.execute("SELECT * FROM elearningapp.lo")
        lo_list = result.mappings().all()
        print(f"Load {len(lo_list)} learning objects")
        return lo_list

async def generate_interactions(learner_profile, lo_list):
    return None