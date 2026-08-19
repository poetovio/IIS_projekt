<script lang="ts">
	const API_URL = 'http://127.0.0.1:8000';

	let zones = $state([
		'AT',
		'BE',
		'BG',
		'CH',
		'CZ',
		'DE_LU',
		'DK1',
		'DK2',
		'EE',
		'ES',
		'FI',
		'FR',
		'GR',
		'HR',
		'HU',
		'IE',
		'IT_NORD',
		'IT_CNOR',
		'IT_CSUD',
		'IT_SUD',
		'LT',
		'LU',
		'LV',
		'ME',
		'NL',
		'NO1',
		'NO2',
		'NO3',
		'NO4',
		'NO5',
		'PL',
		'PT',
		'RO',
		'SE1',
		'SE2',
		'SE3',
		'SE4',
		'SI',
		'SK'
	]);

	let selectedZone = $state('SI');
	let predicting = $state(false);
	let prediction = $state<any>(null);
	let error = $state('');

	async function predictRegression() {
		predicting = true;
		prediction = null;
		error = '';

		try {
			const dataResponse = await fetch(
				`${API_URL}/data/${selectedZone}`
			);

			if (!dataResponse.ok) {
				const result = await dataResponse.json();
				throw new Error(
					result.detail ?? 'Unable to load market data.'
				);
			}

			const dataResult = await dataResponse.json();

			const input = dataResult.data.map(
				(row: Record<string, unknown>) => [
					Number(row.price),
					Number(row.hour),
					Number(row.day_of_week),
					Number(row.day),
					Number(row.month),
					Number(row.year),
					row.is_weekend ? 1 : 0
				]
			);

			const response = await fetch(
				`${API_URL}/predict/regression/${selectedZone}`,
				{
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify({
						data: input
					})
				}
			);

			if (!response.ok) {
				const result = await response.json();

				throw new Error(
					result.detail ?? 'Prediction failed.'
				);
			}

			const result = await response.json();

			console.log('REGRESSION RESULT:', result);

			prediction = result;
		} catch (err) {
			error =
				err instanceof Error
					? err.message
					: 'Unable to generate prediction.';
		} finally {
			predicting = false;
		}
	}
</script>

<svelte:head>
	<title>Energy Intelligence</title>
	<meta
		name="description"
		content="Intelligent electricity price prediction system"
	/>
</svelte:head>

<div class="min-h-screen bg-slate-950 text-white">
	<header class="border-b border-white/10 bg-slate-950/90">
		<div class="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
			<div class="flex items-center gap-3">
				<div
					class="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-400 font-bold text-slate-950"
				>
					⚡
				</div>

				<div>
					<div class="font-semibold tracking-tight">
						Energy Intelligence
					</div>

					<div class="text-xs text-slate-400">
						Intelligent Energy System
					</div>
				</div>
			</div>

			<a
				href="/admin"
				class="rounded-lg border border-white/10 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-white/20 hover:bg-white/5 hover:text-white"
			>
				Admin Dashboard →
			</a>
		</div>
	</header>

	<main class="mx-auto max-w-7xl px-6 py-12">
		<section class="mb-12">
			<div
				class="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1 text-xs font-medium text-emerald-300"
			>
				<span class="h-2 w-2 rounded-full bg-emerald-400"></span>
				PRODUCTION INTELLIGENCE SYSTEM
			</div>

			<h1 class="max-w-4xl text-4xl font-bold tracking-tight sm:text-6xl">
				Understand electricity prices
				<span class="text-emerald-400">before they move.</span>
			</h1>

			<p class="mt-5 max-w-2xl text-lg leading-8 text-slate-400">
				Use neural-network models to predict and classify electricity
				prices across European energy markets.
			</p>
		</section>

		<section class="grid gap-4 lg:grid-cols-2">
			<div
				class="rounded-2xl border border-white/10 bg-white/[0.04] p-7"
			>
				<div
					class="text-xs font-semibold tracking-widest text-emerald-300"
				>
					REGRESSION
				</div>

				<h2 class="mt-2 text-2xl font-semibold">
					Electricity price forecast
				</h2>

				<p class="mt-4 text-sm leading-7 text-slate-400">
					The LSTM model uses the latest 24 hourly observations to
					predict the next electricity price.
				</p>

				<label
					class="mt-7 block text-sm font-medium text-slate-300"
					for="zone"
				>
					Market zone
				</label>

				<select
					id="zone"
					bind:value={selectedZone}
					class="mt-2 w-full rounded-xl border border-white/10 bg-slate-950 px-4 py-3 text-white outline-none transition focus:border-emerald-400"
					disabled={predicting}
				>
					{#each zones as zone}
						<option value={zone}>
							{zone}
						</option>
					{/each}
				</select>

				<div class="mt-4 grid grid-cols-2 gap-3">
					<div class="rounded-xl bg-black/20 p-4">
						<div class="text-xs text-slate-500">
							MODEL
						</div>

						<div class="mt-1 text-lg font-semibold">
							LSTM
						</div>
					</div>

					<div class="rounded-xl bg-black/20 p-4">
						<div class="text-xs text-slate-500">
							WINDOW
						</div>

						<div class="mt-1 text-lg font-semibold">
							24 hours
						</div>
					</div>
				</div>

				<button
					class="mt-5 w-full rounded-xl bg-emerald-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
					onclick={predictRegression}
					disabled={predicting}
				>
					{#if predicting}
						Generating prediction...
					{:else}
						Predict next price
					{/if}
				</button>

				{#if error}
					<div
						class="mt-4 rounded-xl border border-red-400/20 bg-red-400/5 p-4 text-sm text-red-300"
					>
						{error}
					</div>
				{/if}
			</div>

			<div
				class="rounded-2xl border border-white/10 bg-white/[0.04] p-7"
			>
				<div
					class="text-xs font-semibold tracking-widest text-slate-500"
				>
					PREDICTION
				</div>

				{#if prediction !== null}
					<div class="mt-6">
						<div class="text-sm text-slate-500">
							{prediction.zone} · LSTM Regression
						</div>

						<div class="mt-3 flex items-end gap-3">
							<span class="text-6xl font-bold text-emerald-400">
								{prediction.prediction.toFixed(2)}
							</span>

							<span class="mb-2 text-lg text-slate-400">
								EUR/MWh
							</span>
						</div>
					</div>
				{:else}
					<div
						class="flex min-h-[280px] items-center justify-center"
					>
						<div class="text-center">
							<div class="text-6xl font-light text-slate-700">
								€
							</div>

							<p class="mt-4 text-slate-500">
								Your prediction will appear here.
							</p>
						</div>
					</div>
				{/if}
			</div>
		</section>

		<section
			class="mt-4 rounded-2xl border border-white/10 bg-white/[0.04] p-7"
		>
			<div
				class="text-xs font-semibold tracking-widest text-slate-500"
			>
				INTELLIGENT SYSTEM
			</div>

			<div class="mt-5 grid gap-4 md:grid-cols-3">
				<div class="rounded-xl bg-black/20 p-5">
					<div class="text-emerald-400">01</div>

					<div class="mt-2 font-semibold">
						Market data
					</div>

					<p class="mt-2 text-sm leading-6 text-slate-500">
						Latest electricity price observations are used as
						input for the model.
					</p>
				</div>

				<div class="rounded-xl bg-black/20 p-5">
					<div class="text-emerald-400">02</div>

					<div class="mt-2 font-semibold">
						Neural network
					</div>

					<p class="mt-2 text-sm leading-6 text-slate-500">
						A trained LSTM model analyses the recent price
						sequence.
					</p>
				</div>

				<div class="rounded-xl bg-black/20 p-5">
					<div class="text-emerald-400">03</div>

					<div class="mt-2 font-semibold">
						Prediction
					</div>

					<p class="mt-2 text-sm leading-6 text-slate-500">
						The system returns the predicted electricity price
						for the selected market.
					</p>
				</div>
			</div>
		</section>
	</main>

	<footer class="mt-12 border-t border-white/10">
		<div
			class="mx-auto flex max-w-7xl justify-between px-6 py-6 text-xs text-slate-500"
		>
			<span>Energy Intelligence System</span>
			<span>Neural Network Intelligence</span>
		</div>
	</footer>
</div>