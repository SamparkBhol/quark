import click


@click.group()
def cli():
    pass


@cli.command()
@click.option('--samples', default=1500)
@click.option('--epochs', default=12)
@click.option('--batch', default=32)
@click.option('--qmax', default=4)
@click.option('--lr', default=3e-4)
@click.option('--loss', default='infonce', type=click.Choice(['triplet', 'infonce']))
@click.option('--out', default='quark.pt')
def train(samples, epochs, batch, qmax, lr, loss, out):
    from .train import train as do_train
    do_train(samples=samples, epochs=epochs, batch=batch, qmax=qmax, lr=lr, loss=loss, save_to=out)


@cli.command(name='eval')
@click.option('--weights', default=None)
@click.option('--queries', default=100)
@click.option('--lib', default=500)
@click.option('--equiv', default=2)
@click.option('--out', default=None)
def evalcmd(weights, queries, lib, equiv, out):
    from .eval_ import run
    run(weights=weights, n_queries=queries, lib_size=lib, equiv_per_query=equiv, out=out)


@cli.command()
@click.option('--n', default=3, type=int)
@click.option('--depth', default=10, type=int)
@click.option('--k', default=3, type=int)
@click.option('--seed', default=0, type=int)
def show(n, depth, k, seed):
    from .pairs import make_pair
    from .verify import equiv
    a, b = make_pair(n, depth, k=k, seed=seed)
    print("anchor:")
    print(a.draw(output='text'))
    print()
    print("rewritten (positive):")
    print(b.draw(output='text'))
    print()
    print(f"equivalent up to global phase: {equiv(a, b)}")


@cli.command()
@click.option('--circuits', default=50, type=click.IntRange(min=1))
@click.option('--phys', default=8, type=click.IntRange(min=2))
@click.option('--seed', default=0, type=int)
@click.option('--weights', default=None)
def bench(circuits, phys, seed, weights):
    from .eval_ import smoke_bench
    smoke_bench(n_circuits=circuits, qmax=phys, seed=seed, weights=weights)


@cli.command()
@click.argument('directory')
@click.option('--weights', default='quark.pt')
@click.option('--threshold', default=0.9)
@click.option('--verify/--no-verify', default=False)
@click.option('--out', default=None)
def dedupe(directory, weights, threshold, verify, out):
    from .dedupe import cluster, fmt_report, _walk
    paths = list(_walk(directory))
    if not paths:
        print(f"no .qasm files found in {directory}")
        return
    print(f"scanning {len(paths)} files...")
    res = cluster(paths, weights=weights, threshold=threshold, verify=verify)
    print()
    print(fmt_report(res))
    if out:
        import json as _json
        with open(out, 'w') as f:
            _json.dump({'groups': res['groups'], 'failed': res['failed']}, f, indent=2)


if __name__ == '__main__':
    cli()
