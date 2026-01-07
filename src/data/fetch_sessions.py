import fastf1

fastf1.Cache.enable_cache('data/raw/fastf1_cache')


def fetch_target_session(target_config):
    """ Fetch the target session from the FastF1

    Parameters
    ----------
    target_config : dict

    Returns
    -------
    session : Session
    """
    season = target_config['season']
    round_num = target_config['round']
    session = fastf1.get_session(season, round_num, 'Q')
    session.load(laps=True)
    return session


def fetch_historical_sessions(seasons, features_config):
    """
    Fetch qualifying results and FP times for explicit seasons.
    Returns a list of dicts with session data.

    Parameters
    ----------
    seasons : list
        The list of seasons to take into consideration for historical data.
    features_config : dict
        The config for features data, whether include an FP session or not.

    Returns
    -------
    sessions_list : list[dict]
        List of dicts, each containing 'season', 'round', 'session', 'fp1', 'fp2', 'fp3'.

    Raises
    ------
    ValueError if no sessions were found.
    """
    sessions_list = []

    for year in seasons:
        schedule = get_event_schedule_safe(year)
        if schedule is None:
            continue
        for _, race in schedule.iterrows():
            round_num = race['RoundNumber']
            try:
                q_session = fastf1.get_session(year, round_num, 'Q')
                q_session.load(laps=True)

                row = {'season': year, 'round': round_num, 'session': q_session}

                for fp in ['FP1', 'FP2', 'FP3']:
                    if features_config.get(f'use_{fp.lower()}', False):
                        try:
                            fp_session = fastf1.get_session(year, round_num, fp)
                            fp_session.load(laps=True)
                            row[fp.lower()] = fp_session
                        except Exception as e:
                            print(f"Could not load {fp} for {year} round {round_num}: {e}")
                            row[fp.lower()] = None
                    else:
                        row[fp.lower()] = None

                sessions_list.append(row)
            except Exception as e:
                print(f"Skipping {year} round {round_num} due to: {e}")

    if not sessions_list:
        raise(ValueError("No sessions were found"))

    return sessions_list


def get_event_schedule_safe(year):
    """
    Try to fetch event schedule from FastF1.

    Parameters
    ----------
    year : int
        The year of the event schedule.

    Returns
    -------
    schedule : EventSchedule | None
    """
    try:
        print(f"Loading schedule for year: {year}")
        return fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Skipping season {year}: failed to load schedule ({e})")
        return None
